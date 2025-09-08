import math

import torch
import torch.nn as nn
from utils.tool import *
import ipdb


class ST_GCNN_layer(nn.Module):
    def __init__(
        self,
        in_channels,  # xyz
        out_channels,  # 128
        kernel_size,  # [3, 1]
        stride,  # 1
        time_dim,  # n_dct 20
        joints_dim,  # 20
        dropout,
        version=0,
        pose_info=None,
        SE_block=True,
        se_ctrl=None,
    ):
        super(ST_GCNN_layer, self).__init__()
        self.kernel_size = kernel_size
        assert self.kernel_size[0] % 2 == 1
        assert self.kernel_size[1] % 2 == 1
        padding = ((self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2)
        self.SE_block = SE_block

        if version == 0:
            self.gcn = ConvTemporalGraphical(time_dim, joints_dim)
        elif version == 1:
            self.gcn = ConvTemporalGraphicalV1(time_dim, joints_dim, pose_info=pose_info)

        self.tcn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (self.kernel_size[0], self.kernel_size[1]), (stride, stride), padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(1, 1)), nn.BatchNorm2d(out_channels))
        else:
            self.residual = nn.Identity()

        self.prelu = nn.PReLU()

        if self.SE_block:
            if in_channels <= 16:
                self.SE_block = False
            elif in_channels == 32:
                reduction = 2
            elif in_channels == 64:
                reduction = 8
            elif in_channels == 128:
                reduction = 32
            else:
                raise ValueError(f"in_channels must be 16, 32, 64 or 128, but got {in_channels}")

        if self.SE_block:
            self.SE_spatial = SEBlock(in_channels=in_channels * joints_dim, reduction=reduction * joints_dim)
            self.SE_temporal = SEBlock(in_channels=time_dim, reduction=4)

            # SE modulation to prevent over-constraining features
            if se_ctrl is None:
                w_s, w_t, sd = 0.5, 0.5, 0.1
            else:
                w_s, w_t, sd = se_ctrl
            self.se_weight_spatial = float(w_s)
            self.se_weight_temporal = float(w_t)
            self.se_dropout = nn.Dropout(float(sd)) if float(sd) > 0 else nn.Identity()

            def _init_gate(p: float):
                if p <= 0.0 or p >= 1.0:
                    return torch.tensor(0.0)
                return torch.tensor(math.log(p / (1.0 - p)))

            self.se_gate_spatial_raw = nn.Parameter(_init_gate(self.se_weight_spatial))
            self.se_gate_temporal_raw = nn.Parameter(_init_gate(self.se_weight_temporal))

    def _apply_se_spatial(self, x):
        b, c, dct, j = x.shape
        se_s = self.SE_spatial(x.permute(0, 1, 3, 2).reshape(b, -1, dct)).view(b, -1, j, dct).permute(0, 1, 3, 2)
        return self.se_dropout(se_s)

    def _apply_se_temporal(self, x):
        b, c, dct, j = x.shape
        se_t = self.SE_temporal(x.permute(0, 2, 1, 3).reshape(b, dct, -1)).view(b, dct, -1, j).permute(0, 2, 1, 3)
        return self.se_dropout(se_t)

    def forward(self, x, caption_labels=None):
        """
        x: bs*nk, feature, dct, joint
        """

        res = self.residual(x)

        x = self.gcn(x)
        if self.SE_block:
            gate_s = torch.sigmoid(self.se_gate_spatial_raw)
            x = x + (self.se_weight_spatial * gate_s) * self._apply_se_spatial(x)

        x = self.tcn(x)
        if self.SE_block:
            gate_t = torch.sigmoid(self.se_gate_temporal_raw)
            x = x + (self.se_weight_temporal * gate_t) * self._apply_se_temporal(x)

        x = x + res
        x = self.prelu(x)
        return x


# ========== STARS 16 version ==========
class GTEN_Encoder_16(nn.Module):
    def __init__(
        self,
        cfg,
        input_channels,  # xyz
        joints_to_consider,  # 20
        dct,
        SE_block=True,
        Caption=True,
    ):
        super(GTEN_Encoder_16, self).__init__()
        self.dct_m, self.idct_m = dct
        self.n_pre = cfg["specs"]["n_pre"]

        self.dropout = cfg["dropout"]
        self.input_time_frame = cfg["t_his"]
        self.output_time_frame = cfg["t_pred"]
        self.st_gcnns = nn.ModuleList()

        se_ctrl = (
            cfg.get("se_weight_spatial", 0.5),
            cfg.get("se_weight_temporal", 0.5),
            cfg.get("se_dropout", 0.1),
        )

        param = {
            "kernel_size": [3, 1],
            "stride": 1,
            "time_dim": self.n_pre,
            "joints_dim": joints_to_consider,
            "version": 0,
            "dropout": cfg["dropout"],
            "SE_block": SE_block,
            "se_ctrl": se_ctrl,
        }

        self.st_gcnns.append(ST_GCNN_layer(input_channels, 64, **param))
        self.st_gcnns.append(ST_GCNN_layer(64, 32, **param))
        self.st_gcnns.append(ST_GCNN_layer(32, 16, **param))
        self.st_gcnns.append(ST_GCNN_layer(16, 16, **param))

    def dct_gt(self, gt):
        dct_m = self.dct_m.to(gt.device)
        frame, bs, _ = gt.shape
        T, N, V, C = gt.view(frame, bs, -1, 3).shape  # [t, bs, j, v]

        gt = gt.permute(1, 0, 2)
        inp = torch.matmul(dct_m[: self.n_pre], gt).reshape([N, -1, C, V]).permute(0, 2, 1, 3)  # b c t v

        return inp

    def dct_x(self, x):
        dct_m = self.dct_m.to(x.device)
        # x, y [T, bs, f]
        x = x.view(x.shape[0], x.shape[1], -1, 3)  # T, bs, J, V
        x = x.permute(1, 3, 0, 2)  # bs, V, T, j
        idx_pad = list(range(self.input_time_frame)) + [self.input_time_frame - 1] * self.output_time_frame  # pad last frame
        y = torch.zeros((x.shape[0], x.shape[1], self.output_time_frame, x.shape[3])).to(x.device)
        inp = torch.cat([x, y], dim=2).permute(0, 2, 1, 3)
        N, T, C, V = inp.shape  # bs, his+pre, V, J
        inp = inp.reshape([N, T, C * V])  # [16, 100, 60]
        inp = torch.matmul(dct_m[: self.n_pre], inp[:, idx_pad, :]).reshape([N, -1, C, V]).permute(0, 2, 1, 3)
        return inp

    def forward(self, gt=None, x=None, caption_labels=None):
        # x: [t_h, bs, J*V], gt: [t_h + t_p, bs, J*V]
        inp = self.dct_gt(gt)
        inp = torch.concat([inp, x], 1)
        result = inp

        for st_gcn in self.st_gcnns:
            result = st_gcn(result, caption_labels)

        return result


class Encoder_16(nn.Module):
    def __init__(
        self,
        cfg,
        input_channels,  # xyz
        joints_to_consider,  # 20
        pose_info,
        dct,
        version=[0, 0, 0, 0],
        SE_block=True,
        Caption=True,
    ):
        super(Encoder_16, self).__init__()
        self.dct_m, self.idct_m = dct
        self.nk1, self.nk2 = cfg["nk1"], cfg["nk2"]
        self.n_pre = cfg["specs"]["n_pre"]

        self.dropout = cfg["dropout"]
        self.input_time_frame = cfg["t_his"]
        self.output_time_frame = cfg["t_pred"]
        self.st_gcnns = nn.ModuleList()

        se_ctrl = (
            cfg.get("se_weight_spatial", 0.5),
            cfg.get("se_weight_temporal", 0.5),
            cfg.get("se_dropout", 0.1),
        )

        param = {
            "kernel_size": [3, 1],
            "stride": 1,
            "time_dim": self.n_pre,
            "joints_dim": joints_to_consider,
            "dropout": cfg["dropout"],
            "pose_info": pose_info,
            "SE_block": SE_block,
            "se_ctrl": se_ctrl,
        }

        self.st_gcnns.append(ST_GCNN_layer(input_channels, 128, version=version[0], **param))
        self.st_gcnns.append(ST_GCNN_layer(128, 64, version=version[1], **param))
        self.st_gcnns.append(ST_GCNN_layer(64, 32, version=version[2], **param))
        self.st_gcnns.append(ST_GCNN_layer(32, 16, version=version[3], **param))

        # ------------------------------------------------------------------------------------------------------------------------
        self.e_mu_1 = ST_GCNN_layer(16, 8, **param)
        self.e_mu_2 = nn.Linear(8, 16)
        self.e_logvar_1 = ST_GCNN_layer(16, 8, **param)
        self.e_logvar_2 = nn.Linear(8, 16)

        self.T1 = nn.Parameter(torch.FloatTensor(self.nk1, 1, 32, self.n_pre, 1))
        stdv = 1.0 / math.sqrt(self.T1.size(1))
        self.T1.data.uniform_(-stdv, stdv)

        self.A1 = nn.Parameter(torch.FloatTensor(self.nk2, 1, 32, 1, joints_to_consider))
        stdv = 1.0 / math.sqrt(self.A1.size(1))
        self.A1.data.uniform_(-stdv, stdv)

        self.i1, self.j1 = torch.meshgrid(torch.arange(self.nk1), torch.arange(self.nk2))

    def reparameterize(self, x):
        N, C, T, V = x.shape
        x_mu = self.e_mu_1(x).mean(2).mean(2)
        x_logvar = self.e_logvar_1(x).mean(2).mean(2)
        mu = self.e_mu_2(x_mu).unsqueeze(2).unsqueeze(3).repeat((1, 1, T, V))
        logvar = self.e_logvar_2(x_logvar).unsqueeze(2).unsqueeze(3).repeat((1, 1, T, V))
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std, mu, logvar

    def dct_x(self, x):
        dct_m = self.dct_m.to(x.device)

        # x, y [T, bs, f]
        x = x.view(x.shape[0], x.shape[1], -1, 3)  # T, bs, J, V
        x = x.permute(1, 3, 0, 2)  # bs, V, T, j
        idx_pad = list(range(self.input_time_frame)) + [self.input_time_frame - 1] * self.output_time_frame  # pad last frame
        y = torch.zeros((x.shape[0], x.shape[1], self.output_time_frame, x.shape[3])).to(x.device)
        inp = torch.cat([x, y], dim=2).permute(0, 2, 1, 3)
        N, T, C, V = inp.shape  # bs, his+pre, V, J
        inp = inp.reshape([N, T, C * V])  # [16, 100, 60]
        inp = torch.matmul(dct_m[: self.n_pre], inp[:, idx_pad, :]).reshape([N, -1, C, V]).permute(0, 2, 1, 3)

        return inp

    def forward(self, x, caption_labels=None):
        # x [t_h, bs, J*V] gt [t_h + t_p, bs, J*V]
        dct_m = self.dct_m.to(x.device)

        # x, y [T, bs, f]
        x = x.view(x.shape[0], x.shape[1], -1, 3)  # T, bs, J, V
        x = x.permute(1, 3, 0, 2)  # bs, V, T, j
        idx_pad = list(range(self.input_time_frame)) + [self.input_time_frame - 1] * self.output_time_frame  # pad last frame
        y = torch.zeros((x.shape[0], x.shape[1], self.output_time_frame, x.shape[3])).to(x.device)
        inp = torch.cat([x, y], dim=2).permute(0, 2, 1, 3)
        N, T, C, V = inp.shape  # bs, his+pre, V, J
        inp = inp.reshape([N, T, C * V])  # [16, 100, 60]
        inp = torch.matmul(dct_m[: self.n_pre], inp[:, idx_pad, :]).reshape([N, -1, C, V]).permute(0, 2, 1, 3)
        res = inp
        x = inp

        result = self.st_gcnns[0](inp, caption_labels)
        result = self.st_gcnns[1](result, caption_labels)
        result = self.st_gcnns[2](result, caption_labels)

        result = result + self.T1[self.i1] + self.A1[self.j1]
        result = result.reshape([self.nk1 * self.nk2 * N, -1, self.n_pre, V])

        if caption_labels is not None:
            caption_labels = caption_labels.repeat_interleave(self.nk1 * self.nk2, 0)

        result = self.st_gcnns[3](result, caption_labels)

        return result, res


class Decoder_16(nn.Module):
    def __init__(
        self,
        cfg,
        input_channels,  # xyz
        joints_to_consider,  # 20
        pose_info,
        dct,
        version=[0, 0, 0, 0],
        SE_block=True,
        Caption=True,
        pretrain=False,
    ):
        super(Decoder_16, self).__init__()
        self.SE_block = SE_block
        self.Caption = Caption
        self.dct_m, self.idct_m = dct

        self.nk1, self.nk2 = cfg["nk1"], cfg["nk2"]
        self.n_pre = cfg["specs"]["n_pre"]
        self.dropout = cfg["dropout"]
        self.input_time_frame = cfg["t_his"]
        self.output_time_frame = cfg["t_pred"]
        self.st_gcnns = nn.ModuleList()

        se_ctrl = (
            cfg.get("se_weight_spatial", 0.5),
            cfg.get("se_weight_temporal", 0.5),
            cfg.get("se_dropout", 0.1),
        )

        param = {
            "kernel_size": [3, 1],
            "stride": 1,
            "time_dim": self.n_pre,
            "joints_dim": joints_to_consider,
            "dropout": cfg["dropout"],
            "pose_info": pose_info,
            "SE_block": SE_block,
            "se_ctrl": se_ctrl,
        }

        self.st_gcnns.append(ST_GCNN_layer(32, 32, version=version[0], **param))
        self.st_gcnns.append(ST_GCNN_layer(32, 64, version=version[1], **param))
        self.st_gcnns.append(ST_GCNN_layer(64, 128, version=version[2], **param))
        self.st_gcnns.append(ST_GCNN_layer(128, input_channels, version=version[3], **param))

        # ------------------------------------------------------------------------------------------------------------------------
        self.T2 = nn.Parameter(torch.FloatTensor(self.nk2, 1, 32, self.n_pre, 1))
        stdv = 1.0 / math.sqrt(self.T2.size(1))
        self.T2.data.uniform_(-stdv, stdv)

        self.A2 = nn.Parameter(torch.FloatTensor(self.nk1, 1, 32, 1, joints_to_consider))
        stdv = 1.0 / math.sqrt(self.A2.size(1))
        self.A2.data.uniform_(-stdv, stdv)

        self.i2, self.j2 = torch.meshgrid(torch.arange(self.nk2), torch.arange(self.nk1))
        self.idx2 = torch.tensor([[jj * self.nk2 + ii for jj in range(self.nk1)] for ii in range(self.nk2)], dtype=torch.long).view(
            self.nk2, self.nk1
        )

    def forward(self, result, res, caption_labels=None):
        idct_m = self.idct_m.to(result.device)
        N, C, T, V = res.shape

        if caption_labels is not None:
            caption_labels = caption_labels.repeat_interleave(self.nk1 * self.nk2, 0)

        result = self.st_gcnns[0](result, caption_labels)

        result = result.reshape([self.nk1 * self.nk2, N, -1, T, V])
        result = result[self.idx2] + self.T2[self.i2] + self.A2[self.j2]
        result = result.reshape([self.nk1 * self.nk2 * N, -1, T, V])

        result = self.st_gcnns[1](result, caption_labels)
        result = self.st_gcnns[2](result, caption_labels)
        result = self.st_gcnns[3](result, caption_labels)

        result += res.repeat(self.nk1 * self.nk2, 1, 1, 1)  # initial input residual

        x = result.permute(0, 2, 1, 3).reshape([self.nk1 * self.nk2 * N, -1, C * V])

        x_re = torch.matmul(idct_m[:, : self.n_pre], x).reshape([self.nk1 * self.nk2, N, -1, C, V])

        x = x_re.permute(2, 1, 0, 4, 3).contiguous().view(-1, res.shape[0], self.nk1 * self.nk2, res.shape[1] * res.shape[3], 1).squeeze(4)
        return x
