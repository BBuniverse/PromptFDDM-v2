import time
import copy

import pickle
import datetime
from tqdm import tqdm
from torch import optim
from utils.torch import *
from utils.logger import *
from utils.metrics import *
from models.EncoderDecoder import *
from utils.valid_angle_check import *
from utils.util import absolute2relative_torch

from models.transformer import EMA
from motion_pred.utils.visualizationGIF import render_animationGIF
import torch.nn.functional as F

import ipdb
import swanlab


class DM_Trainer:
    def __init__(self, cfg, args, STEN, diffusion, unet, pose_prior, dataset, dataset_multi_test, logger, tb_logger):
        super().__init__()

        self.dtype = torch.float32
        torch.set_default_dtype(self.dtype)
        self.device = cfg.device

        self.cfg = cfg
        self.args = args
        self.STEN_cfg = cfg.STEN_cfg
        self.RDGN_cfg = cfg.RDGN_cfg

        self.ref_STEN, self.pose_prior = STEN.to(self.device), pose_prior.to(self.device)

        self.dpo_STEN = copy.deepcopy(self.ref_STEN).to(self.device)
        self.diffusion = diffusion
        self.unet = unet.to(self.device)

        model_cp = pickle.load(open(f"results/{cfg.dataset}_nf_random/models/vae_0025.p", "rb"))
        self.pose_prior.load_state_dict(model_cp["model_dict"])

        self.dct_m = self.ref_STEN.dct_m.to(self.device)

        self.ema = EMA(0.999)
        self.ema_model = copy.deepcopy(self.unet).eval().requires_grad_(False)
        self.ema_setup = (True, self.ema, self.ema_model)

        self.gten_feat_dim = self.RDGN_cfg["features"]

        if hasattr(self.diffusion, "motion_size"):
            self.diffusion.motion_size = self.gten_feat_dim

        self.adapter = nn.Sequential(
            nn.LayerNorm(self.gten_feat_dim),
            nn.Linear(self.gten_feat_dim, self.gten_feat_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.gten_feat_dim, self.gten_feat_dim),
            nn.LayerNorm(self.gten_feat_dim),
        ).to(self.device)

        self.dataset_train = dataset["train"]
        self.dataset_test = dataset["test"]
        self.dataset_multi_test = dataset_multi_test
        self.stats = 0
        self.logger = logger
        self.tb_logger = tb_logger

        self.t_his = self.STEN_cfg["t_his"]
        self.t_pred = self.STEN_cfg["t_pred"]
        self.nk = self.STEN_cfg["nk"]

        self.n_modality = 10
        self.DM_epoch = 100
        self.epoch = 0
        self.batch_size = self.RDGN_cfg["batch_size"]

        self.lambda_dpo_warmup_epochs = 50
        self.dpo_jl_scale = 2.0
        self.cond_noise_std_base = 1e-2
        self.cond_noise_warmup_epochs = 10
        self.align_weight_base = 0.8
        self.align_warmup_epochs = 30
        self.align_every_n = 1
        self.consistency_weight = 0.2
        self.global_step = 0

    def _adapt(self, z):
        return self.adapter(z) if hasattr(self, "adapter") and self.adapter is not None else z

    def _get_lambda_dpo(self):
        progress = max(0, self.epoch - self.DM_epoch)
        warm = self.lambda_dpo_warmup_epochs if self.lambda_dpo_warmup_epochs > 0 else 1
        return min(1.0, 0.1 + 0.9 * (progress / warm))

    def _get_align_weight(self):
        if self.epoch < self.align_warmup_epochs:
            progress = (self.epoch + 1) / max(1, self.align_warmup_epochs)
            return self.align_weight_base * progress
        return self.align_weight_base  # maintain high weight throughout training

    def _get_beta_dpo(self):
        if self.epoch < self.lambda_dpo_warmup_epochs:
            return 0.1 + 0.4 * (self.epoch + 1) / max(1, self.lambda_dpo_warmup_epochs)
        return 0.5

    @staticmethod
    def _standardize(x: torch.Tensor):
        eps = 1e-6
        return (x - x.mean()) / (x.std() + eps)

    def _get_caption_labels(self, model, captions):
        if not self.args.Caption or captions is None:
            return None
        if isinstance(captions, (list, tuple)):
            captions_tensor = torch.cat([c for c in captions], dim=0)
        else:
            captions_tensor = captions
        return model.lm_head(captions_tensor.to(self.device))

    def _random_subsample_traj(self, pred_traj):
        ran = np.random.uniform()
        if ran > 0.67:
            traj_tmp = pred_traj[self.t_his :: 3].reshape([-1, pred_traj.shape[-1] // 3, 3])
        elif ran > 0.33:
            traj_tmp = pred_traj[self.t_his + 1 :: 3].reshape([-1, pred_traj.shape[-1] // 3, 3])
        else:
            traj_tmp = pred_traj[self.t_his + 2 :: 3].reshape([-1, pred_traj.shape[-1] // 3, 3])
        tmp = torch.zeros_like(traj_tmp[:, :1, :])
        traj_tmp = torch.cat([tmp, traj_tmp], dim=1)
        traj_tmp = absolute2relative_torch(traj_tmp, parents=self.dataset_train.skeleton.parents()).reshape([-1, pred_traj.shape[-1]])
        return traj_tmp

    def _neg_recon_score(self, pred_seq, Y_gt_fut):
        Yp = pred_seq[self.t_his :]
        diff = Yp - Y_gt_fut.unsqueeze(2)  # [T_fut, B, nk, F]
        dist = diff.pow(2).sum(dim=-1).sum(dim=0)  # [B, nk]
        value, _ = dist.min(dim=1)  # [B]
        return -value  # higher is better

    # ==================== RDGN loop ====================
    def loop(self):
        self.diff_before_train()
        for self.epoch in range(self.diff_start_epoch, self.num_epoch):
            if self.epoch < self.DM_epoch:  # train DM only
                self.diff_before_train_DM_epoch()
                # self.before_val_step()
                # self.run_val_step()
                self.diff_run_train_DM_epoch()
                self.diff_after_train_DM_epoch()
            else:  # finetune STEN with DPO
                self.diff_before_train_joint_epoch()
                self.diff_run_train_joint_epoch_DPO()
                self.diff_after_train_joint_epoch()

            if (self.epoch + 1) % self.RDGN_cfg["save_model_interval"] == 0:  # model saving period
                self.before_val_step()
                self.run_val_step()

        print(self.cfg.cfg["name"], " finished DM training")

    def diff_before_train_DM_epoch(self):
        self.unet.train()
        self.ref_STEN.eval()
        self.criterion = nn.MSELoss()

        self.train_losses = 0
        self.train_grad = 0
        self.total_num_sample = 0
        self.loss_names = ["dm/noise", "dm/align", "dm/consistency"]

        self.generator_train = self.dataset_train.sampling_generator(
            num_samples=self.RDGN_cfg["num_data_sample"], batch_size=self.RDGN_cfg["batch_size"], n_modality=self.n_modality
        )

        self.prior = torch.distributions.Normal(
            torch.tensor(0, dtype=self.dtype, device=self.device), torch.tensor(1, dtype=self.dtype, device=self.device)
        )
        self.time_start = time.time()

    def diff_before_train_joint_epoch(self):
        self.dpo_STEN.train()

        self.unet.train().requires_grad_(True)
        self.ref_STEN.eval().requires_grad_(False)
        self.criterion = nn.MSELoss()

        self.train_losses = 0
        self.train_grad = 0
        self.total_num_sample = 0
        self.loss_names = [
            "dm/LOSS",
            "dm/limb",
            "dm/ang",
            "dm/DIV",
            "dm/Div",
            "dm/pred",
            "dm/his",
            "dm/multi",
            "dm/ADE",
            "p(z)",
            "logdet",
        ]
        self.loss_names.insert(0, "sten/noise")
        if self.args.DPO:
            self.loss_names = ["sten" + name[2:] if name.startswith("dm") else name for name in self.loss_names]
            self.loss_names.insert(0, "sten/DPO")

        self.stats = 0

        self.generator_train = self.dataset_train.sampling_generator(
            num_samples=self.STEN_cfg["num_data_sample"], batch_size=self.STEN_cfg["batch_size"], n_modality=self.n_modality
        )

        self.prior = torch.distributions.Normal(
            torch.tensor(0, dtype=self.dtype, device=self.device), torch.tensor(1, dtype=self.dtype, device=self.device)
        )
        self.time_start = time.time()

    def diff_run_train_DM_epoch(self):
        # Config train DM only
        for traj_np, _, _, captions in tqdm(self.generator_train):
            with torch.no_grad():
                bs, _, nj, _ = traj_np[..., 1:, :].shape
                traj_np = traj_np[..., 1:, :].reshape(traj_np.shape[0], traj_np.shape[1], -1)
                traj = tensor(traj_np, device=self.device, dtype=self.dtype).permute(1, 0, 2).contiguous()
                X = traj[: self.t_his]

            dct_x = self.ref_STEN.encoder.dct_x(x=X)  # [B*nk,16,dct,J], [B,3,DCT,J]

            gt_feat_full = self.ref_STEN.GTEN(gt=traj, x=dct_x)  # [B, 16, 20, 20]
            B, C, DCT, J = gt_feat_full.shape  # B, 16, 20, 20

            gt_feat_tokens = gt_feat_full.permute(0, 2, 1, 3).contiguous()  # [B, 20, 16, 20]
            gt_feat_tokens = gt_feat_tokens.reshape(B, DCT, C * J)  # [B, 20, 320]

            # UNet learns to denoise complete features [B, DCT, C*J]
            t = self.diffusion.sample_timesteps(B).to(self.device)
            gt_feature_t, noise = self.diffusion.noise_motion(gt_feat_tokens, t)

            mod_train = self.RDGN_cfg.get("mod_train", 1.0)
            x_his_train = None if (np.random.random() > mod_train) else dct_x
            # Cond-GTEN inside UNet will process to high-level condition if provided
            pred_noise = self.unet(gt_feature_t, t, x_his=x_his_train)

            loss_noise = self.criterion(noise, pred_noise)

            # Simplified semantic alignment: UNet directly predicts GTEN features
            loss_align = torch.tensor(0.0, device=self.device)
            loss_consistency = torch.tensor(0.0, device=self.device)

            if (self.global_step % self.align_every_n) == 0:
                with torch.no_grad():
                    # Sample from EMA UNet using dct_x as condition -> UNet 内置 Cond-GTEN 处理
                    denoiser = self.ema_model if self.ema_setup[0] else self.unet
                    z0 = self.diffusion.sample_ddim(denoiser, dct_x)  # [B, 20, 320]

                # 1. Direct feature alignment (UNet output vs GT GTEN features)
                refined_features = self.adapter(z0.reshape(-1, z0.shape[-1]))  # [B*20, 320]
                refined_features = refined_features.reshape(z0.shape[0], z0.shape[1], -1)  # [B, 20, 320]

                # Align refined features with GT GTEN features
                loss_align = F.mse_loss(refined_features, gt_feat_tokens.detach())

                # 2. Temporal consistency (pooled features should be similar)
                refined_pooled = refined_features.mean(dim=1)  # [B, 320]
                gt_pooled = gt_feat_tokens.detach().mean(dim=1)  # [B, 320]
                loss_consistency = 1 - F.cosine_similarity(refined_pooled, gt_pooled, dim=-1).mean()

            self.unet_optimizer.zero_grad()

            # Combine loss terms with progressive weighting
            align_weight = self._get_align_weight()
            total_loss = loss_noise + align_weight * loss_align + self.consistency_weight * loss_consistency

            total_loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), max_norm=1.0)

            losses = np.array([loss_noise.item(), loss_align.item(), loss_consistency.item()])
            self.train_losses += losses
            self.total_num_sample += 1
            self.unet_optimizer.step()

            # update EMA after optimizer step to track updated weights; start from step 0
            args_ema, ema, ema_model = self.ema_setup[0], self.ema_setup[1], self.ema_setup[2]
            if args_ema is True:
                ema.step_ema(ema_model, self.unet, step_start_ema=0)

            del traj_np, traj, loss_noise, pred_noise, noise, gt_feature_t
            self.global_step += 1

    def diff_run_train_joint_epoch_DPO(self):
        # Config DPO train
        for traj_np, traj_multimodal_np, _, captions in tqdm(self.generator_train):
            with torch.no_grad():
                bs, _, nj, _ = traj_np[..., 1:, :].shape
                traj_np = traj_np[..., 1:, :].reshape(traj_np.shape[0], traj_np.shape[1], -1)  # n t vc
                traj = tensor(traj_np, device=self.device, dtype=self.dtype).permute(1, 0, 2).contiguous()
                X = traj[: self.t_his]

                traj_multimodal_np = traj_multimodal_np[..., 1:, :]  # [bs, modality, time, nj, xyz]
                traj_multimodal_np = traj_multimodal_np.reshape([bs, self.n_modality, self.t_his + self.t_pred, -1]).transpose([2, 0, 1, 3])
                traj_multimodal = tensor(traj_multimodal_np, device=self.device, dtype=self.dtype)  # .permute(0, 2, 1).contiguous()

            # dct_gt = self.ref_STEN.dct_gt(traj)
            encoded_x, dct_x = self.dpo_STEN.encoder(x=X)
            B = X.shape[1]
            ref_encoded_x, ref_dct_x = self.ref_STEN.encoder(x=X)

            # ----- DM noise-pred supervised loss (enable UNet backprop in DPO) -----
            gt_feat_full_sup = self.ref_STEN.GTEN(gt=traj, x=dct_x)  # [B, 16, 20, 20]
            _, Csup, DCTsup, Jsup = gt_feat_full_sup.shape
            gt_feat_tokens_sup = gt_feat_full_sup.permute(0, 2, 1, 3).reshape(B, DCTsup, Csup * Jsup)  # [B, 20, 320]
            t_sup = self.diffusion.sample_timesteps(B).to(self.device)
            z_noisy_sup, noise_sup = self.diffusion.noise_motion(gt_feat_tokens_sup, t_sup)
            pred_noise_sup = self.unet(z_noisy_sup, t_sup, x_his=dct_x)
            loss_dm = self.criterion(noise_sup, pred_noise_sup)

            # ----- diffusion part -----
            # Get complete GTEN features for DPO comparison
            gt_feat_full = self.ref_STEN.GTEN(gt=traj, x=dct_x)  # [B, 16, 20, 20]
            B, C, DCT, J = gt_feat_full.shape  # B, 16, 20, 20
            # Teacher GTEN pooled feature to [B*nk,16]
            gt_feature = gt_feat_full.mean(dim=(-2, -1)).repeat(self.nk, 1)

            with torch.no_grad():
                denoiser = self.ema_model if self.ema_setup[0] else self.unet
                # UNet outputs complete GTEN features using dct_x condition
                z_tokens = self.diffusion.sample_ddim(denoiser, dct_x)  # [B, 20, 320]

                # Convert to STEN compatibility: [B,20,320] -> [B,C,J,T] -> pool -> [B,C] -> repeat to [B*nk,C]
                refined_z = self.adapter(z_tokens.reshape(-1, z_tokens.shape[-1]))
                refined_z = refined_z.reshape(z_tokens.shape[0], z_tokens.shape[1], -1)
                C = self.STEN_cfg["z_dim"]
                refined_4d = refined_z.view(refined_z.shape[0], refined_z.shape[1], C, -1).permute(0, 2, 1, 3)
                pooled = refined_4d.mean(dim=(-2, -1))  # [B,C]
                gt_feature_vec = pooled.unsqueeze(1).repeat(1, self.nk, 1).view(-1, pooled.shape[-1])
            # ----- diffusion part -----

            ref_caption_labels = self._get_caption_labels(self.ref_STEN, captions)
            policy_caption_labels = self._get_caption_labels(self.dpo_STEN, captions)
            pred_traj = self.dpo_STEN.combine(encoded_x, dct_x, gt_feature_vec, policy_caption_labels)

            # dpo model: use interpretable per-sample scores (negative reconstruction error)
            Y_gt_fut = traj[self.t_his :]

            # Features are already compressed to z_dim, ready for STEN
            pred_plus = self.dpo_STEN.combine(encoded_x, dct_x, gt_feature, policy_caption_labels)
            pred_minus = self.dpo_STEN.combine(encoded_x, dct_x, gt_feature_vec, policy_caption_labels)
            log_pi_plus = self._neg_recon_score(pred_plus, Y_gt_fut)
            log_pi_minus = self._neg_recon_score(pred_minus, Y_gt_fut)

            with torch.no_grad():
                ref_plus = self.ref_STEN.combine(ref_encoded_x, ref_dct_x, gt_feature, ref_caption_labels)
                ref_minus = self.ref_STEN.combine(ref_encoded_x, ref_dct_x, gt_feature_vec, ref_caption_labels)
                log_ref_plus = self._neg_recon_score(ref_plus, Y_gt_fut)
                log_ref_minus = self._neg_recon_score(ref_minus, Y_gt_fut)

            beta = self._get_beta_dpo()
            gamma = 0.0
            # standardize scores for numerical stability
            pi_plus, pi_minus = self._standardize(log_pi_plus), self._standardize(log_pi_minus)
            ref_plus, ref_minus = self._standardize(log_ref_plus), self._standardize(log_ref_minus)
            log_ratios = (pi_plus - pi_minus) - (ref_plus - ref_minus)
            logits = beta * (log_ratios - gamma)
            loss_dpo = -1 * F.logsigmoid(logits).mean()

            # to save computation
            ran = np.random.uniform()
            if ran > 0.67:
                traj_tmp = pred_traj[self.t_his :: 3].reshape([-1, pred_traj.shape[-1] // 3, 3])
                tmp = torch.zeros_like(traj_tmp[:, :1, :])
                traj_tmp = torch.cat([tmp, traj_tmp], dim=1)
                traj_tmp = absolute2relative_torch(traj_tmp, parents=self.dataset_train.skeleton.parents()).reshape([-1, pred_traj.shape[-1]])
            elif ran > 0.33:
                traj_tmp = pred_traj[self.t_his + 1 :: 3].reshape([-1, pred_traj.shape[-1] // 3, 3])
                tmp = torch.zeros_like(traj_tmp[:, :1, :])
                traj_tmp = torch.cat([tmp, traj_tmp], dim=1)
                traj_tmp = absolute2relative_torch(traj_tmp, parents=self.dataset_train.skeleton.parents()).reshape([-1, pred_traj.shape[-1]])
            else:
                traj_tmp = pred_traj[self.t_his + 2 :: 3].reshape([-1, pred_traj.shape[-1] // 3, 3])
                tmp = torch.zeros_like(traj_tmp[:, :1, :])
                traj_tmp = torch.cat([tmp, traj_tmp], dim=1)
                traj_tmp = absolute2relative_torch(traj_tmp, parents=self.dataset_train.skeleton.parents()).reshape([-1, pred_traj.shape[-1]])
            z, prior_logdetjac = self.pose_prior(traj_tmp)
            prior_lkh = self.prior.log_prob(z).sum(dim=-1)

            loss_GTEN, losses, stat = self.loss_function(
                pred_traj, traj, traj_multimodal, prior_lkh, prior_logdetjac, (self.epoch - self.DM_epoch) / self.num_epoch
            )
            self.stats += stat
            # ==================== STEN part ====================

            # zero grads for both optimizers
            self.DPO_STEN_optimizer.zero_grad()
            self.unet_optimizer.zero_grad()
            # DPO weight warmup to avoid early collapse
            lambda_dpo = self._get_lambda_dpo()
            # include DM supervised loss so UNet also learns during DPO
            total_loss = loss_GTEN + lambda_dpo * loss_dpo + loss_dm
            total_loss.backward()
            losses = np.append(np.array([loss_dpo.item(), loss_dm.item()]), losses)
            self.train_losses += losses
            self.total_num_sample += 1
            # clip and step UNet with its own optimizer to preserve LR continuity
            torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)
            self.unet_optimizer.step()
            self.DPO_STEN_optimizer.step()

            del traj_np, traj

    def diff_after_train_DM_epoch(self):
        self.unet_lr_scheduler.step()
        self.train_losses /= self.total_num_sample
        assert len(self.loss_names) == len(self.train_losses), f"loss_names: {self.loss_names}, train_losses: {self.train_losses}"

        losses_str = " ".join(["{}: {:4f}".format(x, y) for x, y in zip(self.loss_names, self.train_losses)])

        for name, loss in zip(self.loss_names, self.train_losses):
            self.tb_logger.add_scalars(name, {"train": loss}, self.epoch)

        eta_sec = (time.time() - self.time_start) * (self.DM_epoch - self.epoch)
        message = f"INFO: [{self.cfg.cfg['name'][:4]}..] epoch: {self.epoch:3d} "
        message += f"[eta: {str(datetime.timedelta(seconds = int(eta_sec)))}, lr:({self.unet_optimizer.param_groups[0]['lr']:.3e})] {losses_str}"

        self.logger.info(message)

        # SwanLab
        metrics = {"lr": float(self.unet_optimizer.param_groups[0]["lr"])}
        for name, loss in zip(self.loss_names, self.train_losses):
            metrics[name] = float(loss)
        swanlab.log(metrics, step=self.epoch)

    def diff_after_train_joint_epoch(self):
        self.lr_scheduler.step()
        self.train_losses /= self.total_num_sample
        assert len(self.loss_names) == len(self.train_losses)

        losses_str = " ".join(["{}: {:.4f}".format(x, y) for x, y in zip(self.loss_names, self.train_losses)])

        for name, loss in zip(self.loss_names, self.train_losses):
            self.tb_logger.add_scalars(name, {"train": loss}, self.epoch - self.DM_epoch)

        eta_sec = (time.time() - self.time_start) * (self.num_epoch - self.epoch)
        message = f"INFO: [{self.cfg.cfg['name'][:4]}..] epoch: {self.epoch:3d} "

        message += f"[eta: {str(datetime.timedelta(seconds = int(eta_sec)))}, lr:({self.DPO_STEN_optimizer.param_groups[0]['lr']:.3e})] {losses_str}"
        message += "\nbranch_stats: " + ", ".join([f"{x:7.4f}" for x in self.stats.tolist()])
        self.stats = 0

        self.logger.info(message)

        # SwanLab
        metrics = {"lr": float(self.DPO_STEN_optimizer.param_groups[0]["lr"])}
        for name, loss in zip(self.loss_names, self.train_losses):
            metrics[name] = float(loss)
        swanlab.log(metrics, step=self.epoch)

    # ==================== RDGN loop ====================

    def eval(self):
        self.logger.info(f"{'='*40} STEN Guide by RDGN{' with prompt' if self.args.promptFlag else ''} {'='*40}")
        self.before_val_step()
        self.run_val_step()

    def before_val_step(self):
        if self.epoch != 0:
            with to_cpu(self.unet):
                cp_path = self.cfg.RDGN_path % ("prompt" if self.args.promptFlag else "", (self.epoch + 1), self.args.STEN_model)
                model_cp = {
                    "denoise_dict": self.ema_model.state_dict() if self.ema_setup[0] else self.unet.state_dict(),
                    "adapter_dict": self.adapter.state_dict(),
                    "ref_STEN_dict": self.ref_STEN.state_dict(),
                    "dpo_STEN_dict": self.dpo_STEN.state_dict(),
                    "denoise_optimizer": self.unet_optimizer.state_dict(),
                    "denoise_scheduler": self.unet_lr_scheduler.state_dict(),
                    "epoch": self.epoch + 1,
                    "meta": {"std": self.dataset_train.std, "mean": self.dataset_train.mean},
                }
                pickle.dump(model_cp, open(cp_path, "wb"))

        self.stats_func = {
            "APD  ": compute_diversity,
            "AMSE ": compute_amse,
            "FMSE ": compute_fmse,
            "ADE  ": compute_ade,
            "FDE  ": compute_fde,
            "MMADE": compute_mmade,
            "MMFDE": compute_mmfde,
            "MPJPE": mpjpe_error,
        }
        self.stats_names = list(self.stats_func.keys())
        self.stats_meter = {x: AverageMeter() for x in self.stats_names}

        self.generator_test = self.dataset_test.iter_generator(step=self.t_his)
        self.ref_STEN.eval()
        self.dpo_STEN.eval()
        self.unet.eval()

    def run_val_step(self):
        num_samples = 0
        num_seeds = 1

        with torch.no_grad():
            for i, (data, _, _, captions) in tqdm(enumerate(self.generator_test)):
                num_samples += 1
                gt = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)[:, self.t_his :, :]
                gt_multi = self.dataset_multi_test[i]
                if gt_multi.shape[0] == 1:
                    continue
                pred = self.get_prediction(data, captions, sample_num=1, num_seeds=num_seeds, concat_hist=False)
                for stats in self.stats_names[:8]:
                    val = 0
                    branches = 0
                    for pred_i in pred:
                        # sample_num * total_len * ((num_key-1)*3), 1 * total_len * ((num_key-1)*3)
                        v = self.stats_func[stats](pred_i, gt, gt_multi)
                        val += v[0] / num_seeds
                        if self.stats_func[stats](pred_i, gt, gt_multi)[1] is not None:
                            branches += v[1] / num_seeds
                    self.stats_meter[stats].update(val)

            self.logger.info("=" * 80)
            for stats in self.stats_names:
                self.logger.info(f"Total {stats}: {self.stats_meter[stats].avg:.3f}")

            self.logger.info("=" * 80)

        # SwanLab
        if self.args.mode == "train":
            val_metrics = {f"dm_val/{k.strip()}": float(v.avg) for k, v in self.stats_meter.items()}
            swanlab.log(val_metrics, step=self.epoch)

    # ==================== RDGN start ====================
    def diff_before_train(self):
        self.num_epoch = (self.DM_epoch + self.STEN_cfg["num_epoch"]) if self.args.DPO else self.STEN_cfg["num_epoch"] + self.DM_epoch
        self.lambdas = self.cfg.cfg["lambdas"]
        self.valid_ang = pickle.load(open(f"dataset/{self.cfg.cfg['dataset']}_valid_angle.p", "rb"))

        # optimize UNet, adapter and compressor together during DM stage
        self.unet_optimizer = optim.Adam(
            [
                {"params": self.unet.parameters(), "lr": self.RDGN_cfg["lr"]},
                {"params": self.adapter.parameters(), "lr": self.RDGN_cfg["lr"] * 0.5},
            ]
        )
        self.unet_lr_scheduler = optim.lr_scheduler.MultiStepLR(self.unet_optimizer, milestones=[60, 90], gamma=0.5)

        dpo_lr = 3e-4 if self.args.DPO else self.STEN_cfg["lr"]
        self.DPO_STEN_optimizer = optim.Adam(
            [
                {"params": self.dpo_STEN.parameters(), "lr": dpo_lr},
                {"params": self.adapter.parameters(), "lr": dpo_lr * 0.5},
            ]
        )

        self.lr_scheduler = get_scheduler(
            self.DPO_STEN_optimizer,
            policy="warmup" if self.args.DPO else "lambda",
            nepoch_fix=10 if self.args.DPO else self.STEN_cfg["num_epoch_fix"],  # warmup for DPO
            nepoch=100 if self.args.DPO else self.STEN_cfg["num_epoch"],
        )
        states = os.listdir(os.path.dirname(self.cfg.RDGN_path))
        if self.args.resume and len(states):
            states.sort()
            print("!!! resume state .. ", states)
            max_state_file = 100

            resume_state = self.cfg.RDGN_path % ("prompt" if self.args.promptFlag else "", max_state_file, self.args.STEN_model)

            resume_model = pickle.load(open(resume_state, "rb"))

            self.unet.load_state_dict(resume_model["denoise_dict"])
            if "adapter_dict" in resume_model:
                self.adapter.load_state_dict(resume_model["adapter_dict"])
            self.ref_STEN.load_state_dict(resume_model["ref_STEN_dict"])
            self.unet_optimizer.load_state_dict(resume_model["denoise_optimizer"])
            self.unet_lr_scheduler.load_state_dict(resume_model["denoise_scheduler"])
            self.diff_start_epoch = resume_model["epoch"]
            self.logger.info(f"Resuming RDGN training from epoch: {resume_model['epoch']}")
        else:
            self.diff_start_epoch = 0

    # ==================== RDGN end ====================

    # ==================== Loss function start ====================
    def recon_loss(self, Y_pred, Y_gt, Y_mm, Y_h, Y_h_gt):
        """

        @param Y_pred: estimate future pose [future, B, nk, j*xyz]
        @param Y_gt: ground truth future pose [future, B, j*xyz]
        @param Y_mm: multimodal future pose [future, B, multi, j*xyz]
        @param Y_h: estimate past pose [past, bs, nk, j*xyz]
        @param Y_h_gt: ground truth past pose [past, B, j*xyz]
        @return: loss
        """
        stat = torch.zeros(Y_pred.shape[2])  # for nk shape

        # tbw reconstruction loss
        diff = Y_pred - Y_gt.unsqueeze(2)  # calculate the difference between future GT with nk predicted future
        dist = diff.pow(2).sum(dim=-1).sum(dim=0)
        value, indices = dist.min(dim=1)  # the closet prediction index across nk
        loss_recon_pred = value.mean()  # prediction
        for i in range(self.nk):  # Calculate the frequency of nearest matches to the gt at each position across nk
            stat[i] = (indices == i).sum()
        stat /= stat.sum()

        # history reconstruction
        diff = Y_h - Y_h_gt.unsqueeze(2)  # TBMC
        loss_recon_his = diff.pow(2).sum(dim=-1).sum(dim=0).mean()  # STARS

        # history ade
        with torch.no_grad():
            ade = torch.norm(diff, dim=-1).mean(dim=0).min(dim=1)[0].mean()

        diff = Y_pred[:, :, :, None, :] - Y_mm[:, :, None, :, :]
        mask = Y_mm.abs().sum(-1).sum(0) > 1e-6
        dist = diff.pow(2)
        with torch.no_grad():
            zeros = torch.zeros([dist.shape[1], dist.shape[2]], requires_grad=False).to(dist.device)
            zeros.scatter_(dim=1, index=indices.unsqueeze(1).repeat(1, dist.shape[2]), src=zeros + dist.max() - dist.min() + 1)
            zeros = zeros.unsqueeze(0).unsqueeze(3).unsqueeze(4)
            dist += zeros

        dist = dist.sum(dim=-1).sum(dim=0)
        value_2, _ = dist.min(dim=1)
        loss_recon_multi = value_2[mask].mean()
        if torch.isnan(loss_recon_multi):
            loss_recon_multi = torch.zeros_like(loss_recon_pred)

        mask = torch.tril(torch.ones([self.nk, self.nk], device=dist.device)) == 0
        yt = Y_pred.reshape([-1, self.nk, Y_pred.shape[3]]).contiguous()
        pdist = torch.cdist(yt, yt, p=1)[:, mask]

        return loss_recon_pred, loss_recon_his, loss_recon_multi, ade, stat, (-pdist / 100).exp().mean(), pdist.mean()

    def angle_loss(self, y):
        angle_names = list(self.valid_ang.keys())
        y = y.reshape([-1, y.shape[-1]])
        ang_cos = globals()[f"{self.cfg.cfg['dataset']}_valid_angle_check_torch"](y)
        loss = torch.tensor(0, dtype=self.dtype, device=y.device)
        b = 1
        for an in angle_names:
            lower_bound = self.valid_ang[an][0]
            if lower_bound >= -0.98:
                if torch.any(ang_cos[an] < lower_bound):
                    loss += (ang_cos[an][ang_cos[an] < lower_bound] - lower_bound).pow(2).mean()
            upper_bound = self.valid_ang[an][1]
            if upper_bound <= 0.98:
                if torch.any(ang_cos[an] > upper_bound):
                    loss += (ang_cos[an][ang_cos[an] > upper_bound] - upper_bound).pow(2).mean()
        return loss

    def loss_function(self, traj_est, traj, traj_multimodal, prior_lkh, prior_logdetjac, _lambda):
        batch_size = self.STEN_cfg["batch_size"]
        nj = self.dataset_train.traj_dim // 3

        Y_pred = traj_est[self.t_his :]  # T B nk nj*xyz
        Y_gt = traj[self.t_his :]
        Y_multimodal = traj_multimodal[self.t_his :]

        Recon_pred, Recon_his, Recon_mm, ade, stat, JL, div = self.recon_loss(Y_pred, Y_gt, Y_multimodal, traj_est[: self.t_his], traj[: self.t_his])

        # maintain limb length
        parent = self.dataset_train.skeleton.parents()
        tmp = traj[0].reshape([batch_size, nj, 3])
        pgt = torch.zeros([batch_size, nj + 1, 3], dtype=self.dtype, device=self.device)
        pgt[:, 1:] = tmp
        limb_gt = torch.norm(pgt[:, 1:] - pgt[:, parent[1:]], dim=2)[None, :, None, :]
        tmp = traj_est.reshape([-1, batch_size, self.nk, nj, 3])
        pest = torch.zeros([tmp.shape[0], batch_size, self.nk, nj + 1, 3], dtype=self.dtype, device=self.device)
        pest[:, :, :, 1:] = tmp
        limb_est = torch.norm(pest[:, :, :, 1:] - pest[:, :, :, parent[1:]], dim=4)
        loss_limb = torch.mean((limb_gt - limb_est).pow(2).sum(dim=3))

        # angle loss
        loss_ang = self.angle_loss(Y_pred)
        loss_r = (
            loss_limb * self.lambdas[1]
            + JL * self.lambdas[3]
            + Recon_pred * self.lambdas[4]
            + Recon_mm * self.lambdas[5]
            - prior_lkh.mean() * self.lambdas[6]
            + Recon_his * self.lambdas[7]
        )

        if loss_ang > 0:
            loss_r += loss_ang * self.lambdas[8]
        return (
            loss_r,
            np.array(
                [
                    loss_r.item(),
                    loss_limb.item(),
                    loss_ang.item(),
                    JL.item(),
                    div.item(),
                    Recon_pred.item(),
                    Recon_his.item(),
                    Recon_mm.item(),
                    ade.item(),
                    prior_lkh.mean().item(),
                    prior_logdetjac.mean().item(),
                ]
            ),
            stat,
        )

    # ==================== Loss function start ====================

    def get_prediction(self, data, captions=None, sample_num=1, num_seeds=1, concat_hist=True):
        # 1 * total_len * num_key * 3
        bs, T, _, _ = data[..., 1:, :].shape
        traj_np = data[..., 1:, :].reshape(bs, T, -1)
        traj = tensor(traj_np, device=self.device, dtype=self.dtype).permute(1, 0, 2).contiguous()  # [100, 1, 60]

        X = traj[: self.t_his]
        Y_gt = traj[self.t_his :]
        X = X.repeat((1, sample_num * num_seeds, 1))  # [20, 1, 60]
        Y_gt = Y_gt.repeat((1, sample_num * num_seeds, 1))  # [20, 1, 60]

        encoded_x, dct_x = self.dpo_STEN.encoder(x=X)

        # ----- diffusion part -----
        with torch.no_grad():
            denoiser = self.ema_model if (self.ema_setup[0]) else self.unet
            z_tokens = self.diffusion.sample_ddim(denoiser, dct_x)  # [B, 20, D]

            refined_features = self.adapter(z_tokens.reshape(-1, z_tokens.shape[-1]))
            refined_features = refined_features.reshape(z_tokens.shape[0], z_tokens.shape[1], -1)
            refined_4d = refined_features.view(refined_features.shape[0], refined_features.shape[1], self.STEN_cfg["z_dim"], -1).permute(0, 2, 1, 3)
            pooled = refined_4d.mean(dim=(-2, -1))  # [B,C]
            gt_feature_0 = pooled.unsqueeze(1).repeat(1, self.nk, 1).view(-1, pooled.shape[-1])
        # ----- diffusion part -----

        # ----- guiding part -----
        caption_labels = self._get_caption_labels(self.dpo_STEN, captions) if captions is not None else None
        Y = self.dpo_STEN.combine(encoded_x, dct_x, gt_feature_0, caption_labels)
        # ----- guiding part -----

        Y = Y[self.t_his :]  # [80, 1, 10, 60]
        if concat_hist:
            X = X.unsqueeze(2).repeat(1, sample_num * num_seeds, self.nk, 1)
            Y = torch.cat((X, Y), dim=0)

        Y = Y.squeeze(1).permute(1, 0, 2).contiguous().cpu().numpy()
        if Y.shape[0] > 1:
            Y = Y.reshape(-1, self.nk * sample_num, Y.shape[-2], Y.shape[-1])
        else:
            Y = Y[None, ...]
        # num_seeds * sample_num * total_len * feature_size
        return Y  # [bs, nk, t_pre, xyz*joints]

    def visualize(self, cfg):
        self.ref_STEN.eval()
        self.dpo_STEN.eval()
        self.unet.eval()

        def denomarlize(*data):
            out = []
            for x in data:
                x = x * self.dataset_test.std + self.dataset_test.mean
                out.append(x)
            return out

        def post_process(pred, data):
            pred = pred.reshape(pred.shape[0], pred.shape[1], -1, 3)
            if cfg.normalize_data:
                pred = denomarlize(pred)
            pred = np.concatenate((np.tile(data[..., :1, :], (pred.shape[0], 1, 1, 1)), pred), axis=2)
            pred[..., :1, :] = 0
            return pred

        def pose_generator():
            while True:
                data, _, action, captions, fr_start = self.dataset_test.sample(n_modality=10)
                gt = data[0].copy()
                gt[:, :1, :] = 0

                poses = {"action": action, "context": gt, "gt": gt, "fr_start": fr_start}
                with torch.no_grad():
                    pred = self.get_prediction(data, captions, sample_num=1)[0]
                    pred = post_process(pred, data)
                    for i in range(pred.shape[0]):
                        poses[f"{i}"] = pred[i]
                yield poses

        policy_mode = "_prompt" if self.args.promptFlag and self.args.diffusion else ""

        pose_gen = pose_generator()
        for i in tqdm(range(self.args.n_viz), bar_format="{l_bar}{bar:30}{r_bar}", desc="Visualizing"):
            render_animationGIF(
                self.dataset_test.skeleton,
                pose_gen,
                self.t_his,
                output=f"{self.cfg.visualization}/STEN_RDGN_{policy_mode}_{self.args.RDGN_model:03d}_epoch/",
                ncol=2 + self.nk,
                scale=self.cfg.scale,
            )
            # render_animation12(
            #     self.dataset_test.skeleton,
            #     pose_gen,
            #     self.t_his,
            #     output=f"{self.cfg.visualization}/STEN_RDGN_{policy_mode}_{self.args.RDGN_model:03d}_epoch/",
            #     scale=self.cfg.scale,
            # )

            # render_animationNK(
            #     self.dataset_test.skeleton,
            #     pose_gen,
            #     self.t_his,
            #     output=f"{self.cfg.visualization}/STEN_RDGN_{policy_mode}_{self.args.RDGN_model:03d}_epoch/",
            #     scale=self.cfg.scale,
            # )
