import math
import torch
import torch.nn as nn
from utils.tool import *
from models.EncoderDecoder import *
from utils.util import get_dct_matrix
from models.text_model.text_model import LMHead


class STEN(nn.Module):
    """Spatio-Temporal Extractor Network (STEN) and GT Extractor Network (GTEN) for motion prediction."""

    VERSION_CONFIG = {
        "FPHA": [[0, 1, 1, 0], [1, 0, 1, 0]],
        "HO3D": [[0, 0, 0, 0], [0, 0, 0, 0]],
        "H2O": [[0, 0, 0, 0], [0, 0, 0, 0]],
    }
    # Config caption size
    SIZE_PARAMS_CONFIG = {
        "FPHA": {"caption": 16, "gt": 16},
        "HO3D": {"caption": 32, "gt": 32},
        "H2O": {"caption": 4, "gt": 8},
    }

    def __init__(
        self,
        cfg,
        args,
        input_channels,
        joints_to_consider,
        pose_info,
    ):
        super().__init__()
        self.args = args
        self.cfg = cfg.STEN_cfg
        self.latent_dim = self.cfg["z_dim"]
        self.dataset = cfg.cfg["dataset"]

        self.dct_m, self.idct_m = get_dct_matrix(self.cfg["t_his"] + self.cfg["t_pred"])

        encoder_class = globals().get(f"Encoder_{self.latent_dim}")
        decoder_class = globals().get(f"Decoder_{self.latent_dim}")
        gt_encoder_class = globals().get(f"GTEN_Encoder_{self.latent_dim}")

        if not encoder_class or not decoder_class:
            raise ImportError(f"Encoder/Decoder for latent_dim={self.latent_dim} not found in EncoderDecoder.py")

        self.encoder = encoder_class(
            self.cfg,
            input_channels,
            joints_to_consider,
            pose_info,
            (self.dct_m, self.idct_m),
            SE_block=self.args.SE_block,
            version=self.VERSION_CONFIG[self.dataset][0],
        )
        self.decoder = decoder_class(
            self.cfg,
            input_channels,
            joints_to_consider,
            pose_info,
            (self.dct_m, self.idct_m),
            SE_block=self.args.SE_block,
            version=self.VERSION_CONFIG[self.dataset][1],
        )
        self.decoder.st_gcnns[-3].gcn.A = self.encoder.st_gcnns[-1].gcn.A

        size_params = self.SIZE_PARAMS_CONFIG[self.dataset]
        self.lm_head = LMHead()
        self.promptCaption = PromptGenBlock(prompt_dim=self.latent_dim, prompt_size=size_params["caption"])

        self.caption_gate = nn.Parameter(torch.tensor(-6.0))
        self.caption_dropout = nn.Dropout(self.cfg.get("caption_dropout", 0.1))

        cw = self.cfg.get("caption_warmup", 0.3)
        if cw <= 1.0:
            warm = int(round(self.cfg["num_epoch"] * float(cw)))
        else:
            warm = int(cw)
        self.caption_warmup = max(1, warm)

        if self.args.gt:
            print(f"{'='*40} Create GTEN with z={self.latent_dim} {'='*40}")

            self.GTEN = gt_encoder_class(
                self.cfg,
                input_channels=6,
                joints_to_consider=20,
                dct=(self.dct_m, self.idct_m),
                SE_block=self.args.SE_block,
            )

            self.action_weight = nn.Parameter(
                torch.FloatTensor(
                    self.cfg["nk"],
                    self.latent_dim,
                    self.cfg["specs"]["n_pre"],
                    joints_to_consider,
                )
            )
            stdv = 1.0 / math.sqrt(self.action_weight.size(1))
            self.action_weight.data.uniform_(-stdv, stdv)

            self.promptGT = PromptGenBlock(
                prompt_dim=self.latent_dim,
                prompt_size=size_params["gt"],
                lin_dim=self.latent_dim,
            )
            self.gt_anchor = ST_GCNN_layer(
                self.latent_dim * 2,
                self.latent_dim,
                [3, 1],
                1,
                self.cfg["specs"]["n_pre"],
                joints_to_consider,
                dropout=0.1,
                SE_block=False,
            )

        self.current_epoch = 0
        gg = self.cfg.get("gt_grad_warmup", 0.3)
        if gg <= 1.0:
            warm_gt = int(round(self.cfg["num_epoch"] * float(gg)))
        else:
            warm_gt = int(gg)
        self.gt_grad_warmup = max(1, warm_gt)
        self.fuse_gate = nn.Parameter(torch.tensor(-4.0))

    def dct_gt(self, gt):
        dct_m = self.dct_m.to(gt.device)
        frame, bs, _ = gt.shape
        T, N, V, C = gt.view(frame, bs, -1, 3).shape  # [t, bs, j, v]

        gt = gt.permute(1, 0, 2)
        dct_gt_seq = torch.matmul(dct_m[: self.cfg["specs"]["n_pre"]], gt)
        return dct_gt_seq.reshape([N, -1, C, V]).permute(0, 2, 1, 3)  # b c t v

    def fusion(self, x_feature, gt_feature, caption_labels):
        bs, c, dct, j = x_feature.shape

        gt_prompt = self.promptGT(x_feature, gt_feature)

        caption_prompt = self.promptCaption(x_feature, caption_labels.repeat(self.cfg["nk"], 1))
        caption_prompt = self.caption_dropout(caption_prompt)
        gamma = min(1.0, float(self.current_epoch) / float(max(1, self.caption_warmup)))
        caption_prompt = caption_prompt * gamma + caption_prompt.detach() * (1.0 - gamma)
        caption_alpha = torch.sigmoid(self.caption_gate) * gamma
        gt_prompt = gt_prompt + caption_alpha * caption_prompt

        gt_prompt = gt_prompt.reshape(self.cfg["nk"], -1, c, dct, j) + self.action_weight.reshape(self.cfg["nk"], -1, c, dct, j)

        fused = self.gt_anchor(torch.concat([x_feature, gt_prompt.reshape(-1, c, dct, j)], dim=1))

        # gated residual fusion
        alpha = torch.sigmoid(self.fuse_gate)
        x_feature = x_feature + alpha * fused

        return x_feature

    def set_epoch(self, epoch: int, total_epochs: int = None):
        self.current_epoch = int(epoch)
        if total_epochs is not None:
            self.total_epochs = int(total_epochs)

    def forward(self, x, gt, caption_labels):
        """
        Forward pass of the STEN model.

        Args:
            x (torch.Tensor): Input historical motion sequence.
                              Shape: (t_his, bs, n_joints * 3)
            gt (torch.Tensor): Ground truth future motion sequence.
                               Shape: (t_his + t_pred, bs, n_joints * 3)
            caption_labels (torch.Tensor): Text caption labels.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Predicted motion sequence.
                - Mu from the reparameterization.
                - Logvar from the reparameterization.
        """
        caption_labels = self.lm_head(caption_labels)

        encoded_x, dct_x = self.encoder(x)
        x_rand, mu, logvar = self.encoder.reparameterize(encoded_x)

        if self.args.gt:
            gt_feature = self.GTEN(gt=gt, x=dct_x)
            beta = min(1.0, float(self.current_epoch) / float(max(1, self.gt_grad_warmup)))
            gt_feature = gt_feature * beta + gt_feature.detach() * (1.0 - beta)

            gt_feature = gt_feature.repeat(self.cfg["nk"], 1, 1, 1).mean(dim=(-2, -1))  # bs, feature
            encoded_x = self.fusion(encoded_x, gt_feature, caption_labels)

        encoded_x = torch.cat([encoded_x, x_rand], dim=1)

        predicted_seq = self.decoder(encoded_x, dct_x)
        return predicted_seq, mu, logvar

    def combine(self, encoded_x, dct_x, gt_feature, caption_labels=None):
        """
        Combines encoded features with ground truth and caption information.

        Args:
            encoded_x (torch.Tensor): Encoded input sequence.
            dct_x (torch.Tensor): DCT of the input sequence.
            gt_feature (torch.Tensor): Ground truth features.
            caption_labels (torch.Tensor, optional): Text caption labels. Defaults to None.

        Returns:
            torch.Tensor: The combined and decoded sequence.
        """
        x_rand, _, _ = self.encoder.reparameterize(encoded_x)
        encoded_x = self.fusion(encoded_x, gt_feature, caption_labels)

        encoded_x = torch.cat([encoded_x, x_rand], dim=1)
        return self.decoder(encoded_x, dct_x)
