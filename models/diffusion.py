import math
import numpy as np
from utils import *


import ipdb


def sqrt_beta_schedule(timesteps, s=0.0001):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = 1 - torch.sqrt(t + s)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3.0, end=3.0, tau=0.7, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class Diffusion:

    def __init__(
        self,
        cfg,
        beta_start=1e-4,
        beta_end=0.02,
        device="cuda",
        dct_m=None,
        idct_m=None,
    ):
        self.cfg = cfg
        self.noise_steps = cfg["noise_steps"]
        self.beta_start = (1000 / self.noise_steps) * beta_start
        self.beta_end = (1000 / self.noise_steps) * beta_end
        self.motion_size = cfg["features"]
        self.device = device
        self.dct = dct_m.to(self.device)
        self.idct = idct_m.to(self.device)

        self.scheduler = cfg["DM_scheduler"]  # 'Cosine', 'Sqrt', 'Linear', 'Sigmoid'
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.ddim_timesteps = cfg["ddim_timesteps"]

        self.model_type = "data"
        self.EnableComplete = False
        self.mod_enable = True

        self.ddim_timestep_seq = np.asarray(list(range(0, self.noise_steps, self.noise_steps // self.ddim_timesteps))) + 1
        self.ddim_timestep_prev_seq = np.append(np.array([0]), self.ddim_timestep_seq[:-1])

        self.n_pre = cfg["n_pre"]

    def prepare_noise_schedule(self):
        if self.scheduler == "linear":
            return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        elif self.scheduler == "cosine":
            return cosine_beta_schedule(self.noise_steps)
        elif self.scheduler == "sqrt":
            return sqrt_beta_schedule(self.noise_steps)
        elif self.scheduler == "sigmoid":
            return sigmoid_beta_schedule(self.noise_steps)
        else:
            raise NotImplementedError(f"unknown scheduler: {self.scheduler}")

    def noise_motion(self, gt_rand_0, t):
        # Supports both vector inputs [B, D] and token inputs [B, T, D]
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])
        # Broadcast to match input rank
        expand_dims = (1,) * (gt_rand_0.ndim - 1)
        sqrt_alpha_hat = sqrt_alpha_hat.view(-1, *expand_dims)
        sqrt_one_minus_alpha_hat = sqrt_one_minus_alpha_hat.view(-1, *expand_dims)

        Ɛ = torch.randn_like(gt_rand_0)
        return sqrt_alpha_hat * gt_rand_0 + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample_ddim_progressive(self, model, x_cond, noise=None):
        """
        Generate samples from the model and yield samples from each timestep.

        Args are the same as sample_ddim()
        Returns a generator contains x_{prev_t}, shape as [sample_num, n_pre, 3 * joints_num]
        """
        sample_num = len(x_cond)
        if noise is not None:
            x = noise
        else:
            # token-shaped noise: [sample, n_pre, features]
            x = torch.randn((sample_num, self.n_pre, self.motion_size)).to(self.device)

        with torch.no_grad():
            for i in reversed(range(0, self.ddim_timesteps)):
                t = (torch.ones(sample_num) * self.ddim_timestep_seq[i]).long().to(self.device)
                prev_t = (torch.ones(sample_num) * self.ddim_timestep_prev_seq[i]).long().to(self.device)

                # broadcast scalars across token dims
                alpha_hat = self.alpha_hat[t]
                alpha_hat_prev = self.alpha_hat[prev_t]
                expand_dims = (1, 1)
                alpha_hat = alpha_hat.view(-1, *expand_dims)
                alpha_hat_prev = alpha_hat_prev.view(-1, *expand_dims)

                predicted_noise = model(x, t, x_cond)  # Unet

                predicted_x0 = (x - torch.sqrt((1.0 - alpha_hat)) * predicted_noise) / torch.sqrt(alpha_hat)
                pred_dir_xt = torch.sqrt(1 - alpha_hat_prev) * predicted_noise
                x_prev = torch.sqrt(alpha_hat_prev) * predicted_x0 + pred_dir_xt

                x = x_prev

                yield x

    def sample_ddim(self, model, x_cond):
        final = None
        for sample in self.sample_ddim_progressive(model, x_cond):
            final = sample
        return final
