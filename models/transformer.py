import math
import torch
from utils import *
from torch import nn
import torch.nn.functional as F
from models.EncoderDecoder import ST_GCNN_layer

import ipdb


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class FFN(nn.Module):

    def __init__(self, latent_dim, mlp_ratio, dropout, time_embed_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, latent_dim * mlp_ratio)
        self.linear2 = zero_module(nn.Linear(latent_dim * mlp_ratio, latent_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return y


class TemporalSelfAttention(nn.Module):
    def __init__(self, latent_dim, num_head=8, dropout=0.0):
        super().__init__()
        assert latent_dim % num_head == 0, "dim should be divisible by num_heads"

        self.num_heads = num_head
        self.head_dim = latent_dim // num_head
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(latent_dim, latent_dim * 3, bias=False)
        self.proj = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, D]
        B, T, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.permute(0, 2, 1, 3)  # [B, H, T, d]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B,H,T,T]
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        x = attn @ v  # [B,H,T,d]
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        x = self.proj(x)
        return x


class FFN(nn.Module):
    def __init__(self, latent_dim, mlp_ratio, dropout, time_embed_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, latent_dim * mlp_ratio)
        self.linear2 = zero_module(nn.Linear(latent_dim * mlp_ratio, latent_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return y


class PromptGenBlock(nn.Module):
    def __init__(self, prompt_dim=20, prompt_len=10, prompt_size=1024, lin_dim=256):
        super(PromptGenBlock, self).__init__()
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.prompt_param = nn.Parameter(torch.rand(prompt_len, prompt_dim, prompt_size))

    def forward(self, x):
        # x: [c, prompt_len]
        bs = x.shape[0]
        prompt_weights = F.softmax(self.linear_layer(x), dim=1)  # [bs, prompt_len]
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1) * self.prompt_param.repeat(bs, 1, 1, 1).squeeze(1)

        prompt = torch.sum(prompt, dim=1)
        return prompt


class AdaLayerNormZero(nn.Module):
    """
    Norm layer adaptive layer norm zero (adaLN-Zero).
    """

    def __init__(self, embedding_dim, num_embeddings, cond_dim):
        super().__init__()

        self.emb = nn.Linear(num_embeddings, embedding_dim)
        self.cond_proj = nn.Linear(cond_dim, embedding_dim)
        self.embedding_dim = embedding_dim

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, timestep, cond):
        # timestep: [B, num_embeddings]; cond: [B, cond_dim] or [B, T, cond_dim]
        t_proj = self.emb(timestep)  # [B, D]
        if cond is not None:
            c_proj = self.cond_proj(cond)
            if c_proj.dim() == 3:
                t_proj = t_proj.unsqueeze(1).expand(-1, c_proj.size(1), -1)
        else:
            c_proj = 0
        emb_in = t_proj + c_proj
        emb = self.linear(self.silu(emb_in))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=-1)
        x_norm = self.norm(x)
        if scale_msa.dim() == 2:
            x = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        else:
            x = x_norm * (1 + scale_msa) + shift_msa
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class TemporalDiffusionTransformerDecoderLayer(nn.Module):
    def __init__(self, latent_dim=32, time_embed_dim=128, mlp_ratio=256, num_head=4, dct=20, dropout=0.5, prompt=False, cond_dim=16):
        super().__init__()
        self.promptFlag = prompt
        if self.promptFlag:
            self.prompt = PromptGenBlock(prompt_dim=dct, prompt_len=32, prompt_size=latent_dim, lin_dim=768)
            self.promptConv = nn.Conv1d(dct * 2, dct, 1, 1)

        self.adaNormZero = AdaLayerNormZero(latent_dim, time_embed_dim, cond_dim)
        self.norm = nn.LayerNorm(latent_dim, elementwise_affine=False)
        self.sa_block = TemporalSelfAttention(latent_dim, num_head, dropout)
        self.ffn = FFN(latent_dim, mlp_ratio, dropout, time_embed_dim)

    def forward(self, hidden_states, timesteps, cond):
        # hidden_states: [B, T, D], cond: [B, D_cond]
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaNormZero(hidden_states, timesteps, cond)
        if self.promptFlag and cond is not None:
            prompt_inp = cond.mean(dim=1) if (cond is not None and cond.dim() == 3) else cond
            prompt = self.prompt(prompt_inp)
            temp = torch.cat([norm_hidden_states, prompt], dim=1)
            norm_hidden_states = self.promptConv(temp) + norm_hidden_states

        attn_output = self.sa_block(norm_hidden_states)
        if gate_msa.dim() == 2:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        else:
            attn_output = gate_msa * attn_output
        hidden_states = attn_output + hidden_states

        norm_hidden_states = self.norm(hidden_states)
        if scale_mlp.dim() == 2:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        else:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        ff_output = self.ffn(norm_hidden_states)
        if gate_mlp.dim() == 2:
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        else:
            ff_output = gate_mlp * ff_output
        hidden_states = ff_output + hidden_states
        return hidden_states


class MotionTransformer(nn.Module):
    def __init__(self, cfg, dct_m, prompt=False, device="cuda", **kargs):
        super().__init__()
        self.dct_m = dct_m.to(device)
        self.t_his = cfg.STEN_cfg["t_his"]
        self.t_pred = cfg.STEN_cfg["t_pred"]
        cfg_rdgn = cfg.RDGN_cfg
        self.n_pre = cfg_rdgn["n_pre"]  # dct
        self.num_layers = cfg_rdgn["num_layers"]
        self.latent_dim = cfg_rdgn["diff_latent_dim"]
        self.dropout = cfg_rdgn["dropout"]
        self.activation = cfg_rdgn["activation"]

        self.input_feats = cfg_rdgn["features"]
        self.gten_feat_dim = 16 * 20

        self.needs_expansion = self.input_feats != self.gten_feat_dim

        self.time_embed_dim = cfg_rdgn["diff_latent_dim"]
        self.sequence_embedding = nn.Parameter(torch.randn(self.latent_dim))
        self.prompt = prompt

        if self.needs_expansion:
            self.feature_expander = nn.Linear(self.input_feats, self.gten_feat_dim)
            self.joint_embed = nn.Linear(self.gten_feat_dim, self.latent_dim)
        else:
            self.joint_embed = nn.Linear(self.input_feats, self.latent_dim)

        self.temporal_decoder_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            self.temporal_decoder_blocks.append(
                TemporalDiffusionTransformerDecoderLayer(
                    latent_dim=self.latent_dim,
                    time_embed_dim=self.time_embed_dim,
                    mlp_ratio=cfg_rdgn["mlp_ratio"],
                    num_head=cfg_rdgn["num_heads"],
                    dct=self.n_pre,
                    dropout=cfg_rdgn["dropout"],
                    prompt=self.prompt,
                    cond_dim=self.latent_dim,
                )
            )

        self.out = zero_module(nn.Linear(self.latent_dim, self.gten_feat_dim))

        self.encoded_feat_dim = 16  # Feature dimension from STEN encoder
        self.encoded_condition_processor = nn.Sequential(
            nn.Linear(self.encoded_feat_dim, self.latent_dim // 2),
            nn.LayerNorm(self.latent_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.latent_dim // 2, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
        )

        # Cond-GTEN: use ST-GCNN layers to encode dct_x (C=3) into high-level condition
        cond_param = {
            "kernel_size": [3, 1],
            "stride": 1,
            "time_dim": self.n_pre,
            "joints_dim": 20,
            "version": 0,
            "dropout": cfg_rdgn["dropout"],
        }
        # Cond-GTEN path that preserves channel=3 and outputs [B, 3, T, J]
        self.cond_gten_to3 = nn.ModuleList(
            [
                ST_GCNN_layer(3, 64, **cond_param),
                ST_GCNN_layer(64, 32, **cond_param),
                ST_GCNN_layer(32, 3, **cond_param),
            ]
        )
        self.joints_dim = cond_param["joints_dim"]
        # per-token projector: [B, T, 16*J] -> [B, T, latent_dim]
        self.cond_token_projector = nn.Linear(self.gten_feat_dim, self.latent_dim)
        # keep global projector for backward compatibility
        self.cond_gten_processor = nn.Sequential(
            nn.Linear(self.encoded_feat_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
        )
        # flattened projector: [B, 3*T*J] -> [B, latent_dim] (to fuse with timestep embedding)
        self.cond_flatten_projector = nn.Sequential(
            nn.Linear(3 * self.n_pre * self.joints_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
        )

    def forward(self, gt_tokens, timesteps, x_his):
        """
        gt_tokens: [B, T, D] where T=n_pre, D=features (16 or 320)
        x_his:    [B, C, T, J] (DCT condition) OR [B*nk, 16, dct, J] (encodedX condition)
        Returns:  [B, T, 320] - always full GTEN feature dimension
        """
        B, T, D = gt_tokens.shape

        if self.needs_expansion and D != self.gten_feat_dim:
            gt_tokens = self.feature_expander(gt_tokens)  # [B, T, 16] -> [B, T, 320]

        emb = timestep_embedding(timesteps, self.latent_dim)
        h = self.joint_embed(gt_tokens)  # [B, T, 320] -> [B, T, latent_dim]
        h = h + self.sequence_embedding  # [D] broadcast to [B,T,D]

        cond = None
        if x_his is not None:
            if len(x_his.shape) == 4 and x_his.shape[1] == 16:
                cond_e = x_his.mean(dim=(-2, -1))  # -> [B*nk,16]
                if cond_e.shape[0] > B:
                    nk = cond_e.shape[0] // B
                    cond_e = cond_e.view(B, nk, -1).mean(dim=1)
                cond = self.encoded_condition_processor(cond_e)
            else:
                c = x_his
                for layer in self.cond_gten_to3:
                    c = layer(c)
                c_vec = c.view(B, -1)
                cond = self.cond_flatten_projector(c_vec)
        else:
            cond = None

        prelist = []
        for i, module in enumerate(self.temporal_decoder_blocks):
            if i < (self.num_layers // 2):
                prelist.append(h)
                h = module(h, emb, cond)
            else:
                h = module(h, emb, cond)
                h = h + prelist[-1]
                prelist.pop()

        output = self.out(h).contiguous()  # [B, T, latent_dim] -> [B, T, 320]
        return output
