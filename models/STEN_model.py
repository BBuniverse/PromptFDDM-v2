import time

import pickle
import datetime
from tqdm import tqdm
from torch import optim
from utils.torch import *
from utils.logger import *
from utils.metrics import *
from utils.valid_angle_check import *
from utils.util import absolute2relative_torch
from motion_pred.utils.visualizationGIF import render_animationGIF
from motion_pred.utils.visualizationNk import render_animationNK
from motion_pred.utils.visualization12 import render_animation12

import ipdb
import swanlab


class STEN_Trainer:
    def __init__(self, cfg, args, model, diffusion, unet, pose_prior, dataset, dataset_multi_test, logger, tb_logger):
        super().__init__()

        self.dtype = torch.float32
        torch.set_default_dtype(self.dtype)
        self.device = cfg.device

        self.model, self.pose_prior = model.to(self.device), pose_prior.to(self.device)
        model_cp = pickle.load(open(f"results/{cfg.dataset}_nf_random/models/vae_0025.p", "rb"))
        self.pose_prior.load_state_dict(model_cp["model_dict"])

        self.diffusion = diffusion
        self.unet = unet.to(self.device)

        self.dataset_train = dataset["train"]
        self.dataset_test = dataset["test"]
        self.dataset_multi_test = dataset_multi_test
        self.cfg = cfg
        self.args = args
        self.STEN_cfg = cfg.STEN_cfg
        self.RDGN_cfg = cfg.RDGN_cfg
        self.stats = 0
        self.logger = logger
        self.tb_logger = tb_logger

        self.t_his = self.STEN_cfg["t_his"]
        self.t_pred = self.STEN_cfg["t_pred"]
        self.nk = self.STEN_cfg["nk"]

        self.n_modality = 10
        self.epoch = 0

    # ==================== STEN loop ====================
    def loop(self):
        self.before_train()
        for self.epoch in range(self.start_epoch, self.num_epoch):
            self.before_train_step()
            self.run_train_step()
            self.after_train_step()
            if (self.epoch + 1) % self.STEN_cfg["save_model_interval"] == 0:  # model saving period
                self.before_val_step()
                self.run_val_step()

        print(self.cfg.cfg["name"], " finished stage1 training")

    # ==================== STEN loop ====================

    def eval(self):
        policy_mode = "GTEN" if self.args.gt and not self.args.diffusion else "RDGN" if self.args.diffusion else "nothing"
        self.logger.info(f"{'='*40} Guide by {policy_mode}{'_prompt' if self.args.promptFlag else ''} {'='*40}")
        self.before_val_step()
        self.run_val_step()

    # ==================== STEN start ====================
    def before_train(self):
        self.num_epoch = self.STEN_cfg["num_epoch"]
        self.lambdas = self.cfg.cfg["lambdas"]
        self.valid_ang = pickle.load(open(f"dataset/{self.cfg.cfg['dataset']}_valid_angle.p", "rb"))

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.STEN_cfg["lr"])
        self.lr_scheduler = get_scheduler(self.optimizer, policy="lambda", nepoch_fix=self.STEN_cfg["num_epoch_fix"], nepoch=self.num_epoch)
        states = os.listdir(os.path.dirname(self.cfg.STEN_path))

        if self.args.resume and len(states):
            states.sort()
            print("!!! resume state .. ", states)
            max_state_file = max([int(x[5:9]) for x in states])  # STEN_0100.p
            resume_state = self.cfg.STEN_path % (max_state_file)

            resume_model = pickle.load(open(resume_state, "rb"))
            self.model.load_state_dict(resume_model["model_dict"])
            self.optimizer.load_state_dict(resume_model["optimizer"])
            self.lr_scheduler.load_state_dict(resume_model["scheduler"])
            self.start_epoch = resume_model["epoch"]
            self.logger.info(f"Resuming STEN training from epoch: {resume_model['epoch']}")
        else:
            self.start_epoch = 0

    def before_train_step(self):
        self.model.train()
        # pass epoch information for GT gradient warmup and gated fusion
        if hasattr(self.model, "set_epoch"):
            self.model.set_epoch(self.epoch, self.num_epoch)

        self.train_losses = 0
        self.train_grad = 0
        self.total_num_sample = 0
        self.loss_names = [
            "sten/LOSS",
            "sten/limb",
            "sten/ang",
            "sten/DIV",
            "sten/Div",
            "sten/pred",
            "sten/his",
            "sten/multi",
            "sten/ADE",
            "p(z)",
            "logdet",
        ]

        self.generator_train = self.dataset_train.sampling_generator(
            num_samples=self.STEN_cfg["num_data_sample"], batch_size=self.STEN_cfg["batch_size"], n_modality=self.n_modality
        )

        self.prior = torch.distributions.Normal(
            torch.tensor(0, dtype=self.dtype, device=self.device), torch.tensor(1, dtype=self.dtype, device=self.device)
        )

        self.time_start = time.time()

    def run_train_step(self):
        for traj_np, traj_multimodal_np, actions, captions in tqdm(self.generator_train):
            with torch.no_grad():
                bs, t, nj, xyz = traj_np[..., 1:, :].shape
                traj_np = traj_np[..., 1:, :].reshape(traj_np.shape[0], traj_np.shape[1], -1)  # bs t vc
                traj = tensor(traj_np, device=self.device, dtype=self.dtype).permute(1, 0, 2).contiguous()  # t bs vc
                X = traj[: self.t_his]
                Y = traj[self.t_his :]

                traj_multimodal_np = traj_multimodal_np[..., 1:, :]  # [bs, modality, time, nj, xyz]
                traj_multimodal_np = traj_multimodal_np.reshape([bs, self.n_modality, self.t_his + self.t_pred, -1]).transpose([2, 0, 1, 3])
                traj_multimodal = tensor(traj_multimodal_np, device=self.device, dtype=self.dtype)  # .permute(0, 2, 1).contiguous()

            # learning from gt
            captions = torch.concat(captions).to(self.device) if self.args.Caption else None
            pred_traj, _, _ = self.model(X, traj, caption_labels=captions)  # T, bs, nk, f

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
            loss, losses, stat = self.loss_function(pred_traj, traj, traj_multimodal, prior_lkh, prior_logdetjac, self.epoch / self.num_epoch)
            self.stats += stat

            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(list(self.model.parameters()), max_norm=100)
            self.train_grad += grad_norm
            self.optimizer.step()
            self.train_losses += losses
            self.total_num_sample += 1
            del loss, z, pred_traj, traj, traj_np, losses

    def after_train_step(self):
        self.lr_scheduler.step()
        self.train_losses /= self.total_num_sample
        losses_str = " ".join(["{}: {:.4f}".format(x, y) for x, y in zip(self.loss_names, self.train_losses)])
        self.tb_logger.add_scalar("train_grad", self.train_grad / self.total_num_sample, self.epoch)

        for name, loss in zip(self.loss_names, self.train_losses):
            self.tb_logger.add_scalars(name, {"train": loss}, self.epoch)

        eta_sec = (time.time() - self.time_start) * (self.num_epoch - self.epoch)
        message = f"INFO: [{self.cfg.cfg['name'][:4]}..] epoch: {self.epoch:3d} "
        message += f"[eta: {str(datetime.timedelta(seconds = int(eta_sec)))}, lr:({self.optimizer.param_groups[0]['lr']:.3e})] {losses_str}"
        message += "\nbranch_stats: " + ", ".join([f"{x:7.4f}" for x in self.stats.tolist()])
        self.stats = 0

        self.logger.info(message)

        # SwanLab logging
        metrics = {"train/grad": float(self.train_grad / self.total_num_sample), "lr": float(self.optimizer.param_groups[0]["lr"])}
        for name, loss in zip(self.loss_names, self.train_losses):
            metrics[name] = float(loss)
        swanlab.log(metrics, step=self.epoch)

    def before_val_step(self):
        if self.epoch != 0:
            with to_cpu(self.model):
                cp_path = self.cfg.STEN_path % (self.epoch + 1)
                model_cp = {
                    "model_dict": self.model.state_dict(),
                    "gten_dict": self.model.GTEN.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.lr_scheduler.state_dict(),
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
        # stats_names.extend(['ADE_stat', 'FDE_stat', 'MMADE_stat', 'MMFDE_stat', 'MPJPE_stat'])
        self.stats_meter = {x: AverageMeter() for x in self.stats_names}

        self.generator_test = self.dataset_test.iter_generator(step=self.t_his)
        self.model.eval()

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
                        v = self.stats_func[stats](pred_i, gt, gt_multi)
                        val += v[0] / num_seeds
                        if self.stats_func[stats](pred_i, gt, gt_multi)[1] is not None:
                            branches += v[1] / num_seeds
                    self.stats_meter[stats].update(val)

            self.logger.info("=" * 80)
            for stats in self.stats_names:
                self.logger.info(f"Total {stats}: {self.stats_meter[stats].avg:.3f}")

            self.logger.info("=" * 80)

            # SwanLab logging
            val_metrics = {f"val/{k.strip()}": float(v.avg) for k, v in self.stats_meter.items()}
            swanlab.log(val_metrics, step=self.epoch)

    # ==================== STEN end ====================

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
        value_2, indices_2 = dist.min(dim=1)
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
        _lambda = _lambda * 10 if _lambda < 0.1 else 1  # epoch / cfg.num_epoch
        loss_r = (
            loss_limb * self.lambdas[1]
            + JL * self.lambdas[3] * _lambda
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
        # prior_lkh.mean().item(), prior_logdetjac.mean().item(), KLD.item()]), stat

    # ==================== Loss function start ====================

    def get_prediction(self, data, captions=None, sample_num=1, num_seeds=1, concat_hist=True):
        # 1 * total_len * num_key * 3
        bs, T, joints, xyz = data[..., 1:, :].shape
        traj_np = data[..., 1:, :].reshape(bs, T, -1)
        traj = tensor(traj_np, device=self.device, dtype=self.dtype).permute(1, 0, 2).contiguous()  # [100, 1, 60]

        X = traj[: self.t_his]
        Y_gt = traj[self.t_his :]
        X = X.repeat((1, sample_num * num_seeds, 1))  # [20, 1, 60]
        Y_gt = Y_gt.repeat((1, sample_num * num_seeds, 1))  # [20, 1, 60]

        captions = captions.to(self.device) if self.args.Caption else None

        if self.args.gt and not self.args.diffusion:
            Y, _, _ = self.model(X, gt=traj, caption_labels=captions)
        elif self.args.diffusion:  # estimate gt_rand
            encodedX, dct_x = self.model.Encoder(X)  # T, bs, nk, f
            # ----- diffusion part -----
            gt_feature_0 = self.diffusion.sample_ddim(self.unet, dct_x.repeat(self.nk, 1, 1, 1))
            # ----- diffusion part -----

            caption_labels = self.model.lm_head(captions.to(self.device)) if self.args.Caption else None
            Y = self.model.combine(encodedX, dct_x, gt_feature_0, caption_labels)
        else:
            Y, _, _ = self.model(X, gt=None, caption_labels=captions)

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
        self.model.eval()
        if self.args.diffusion:
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
                data, data_multimodal, action, captions, fr_start = self.dataset_test.sample(n_modality=10)
                gt = data[0].copy()
                gt[:, :1, :] = 0

                poses = {"action": action, "context": gt, "gt": gt, "fr_start": fr_start}
                with torch.no_grad():
                    pred = self.get_prediction(data, captions, sample_num=1)[0]
                    pred = post_process(pred, data)
                    for i in range(pred.shape[0]):
                        poses[f"{i}"] = pred[i]
                yield poses

        policy_mode = "GTEN" if self.args.gt and not self.args.diffusion else f"RDGN_{self.args.RDGN_model:04d}" if self.args.diffusion else "None"
        policy_mode += "prompt" if self.args.promptFlag and self.args.diffusion else ""

        pose_gen = pose_generator()
        for i in tqdm(range(self.args.n_viz), bar_format="{l_bar}{bar:30}{r_bar}", desc="Visualizing"):
            render_animationGIF(
                self.dataset_test.skeleton,
                pose_gen,
                self.t_his,
                output=f"{self.cfg.visualization}/STEN_{self.args.STEN_model:03d}_{policy_mode}_epoch/",
                ncol=2 + self.nk,
                scale=self.cfg.scale,
            )

            # render_animation12(
            #     self.dataset_test.skeleton,
            #     pose_gen,
            #     self.t_his,
            #     output=f"{self.cfg.visualization}/STEN_{self.args.STEN_model:03d}_{policy_mode}_epoch/",
            #     scale=self.cfg.scale,
            # )

            # render_animationNK(
            #     self.dataset_test.skeleton,
            #     pose_gen,
            #     self.t_his,
            #     output=f"{self.cfg.visualization}/STEN_{self.args.STEN_model:03d}_{policy_mode}_epoch/",
            #     scale=self.cfg.scale,
            # )
