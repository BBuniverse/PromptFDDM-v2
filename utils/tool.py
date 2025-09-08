import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import ipdb


class ICB(nn.Module):
    def __init__(self, feature_dim, text_dim=384):
        super(ICB, self).__init__()
        self.fc = nn.Linear(text_dim, feature_dim)
        self.beta = nn.Parameter(torch.zeros((1, feature_dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, feature_dim, 1, 1)), requires_grad=True)

    def forward(self, x, text_embedding):
        gating_factors = torch.sigmoid(self.fc(text_embedding))
        gating_factors = gating_factors.unsqueeze(-1).unsqueeze(-1)

        f = x * self.gamma + self.beta  # 1) learned feature scaling/modulation
        f = f * gating_factors  # 2) (soft) feature routing based on text
        return f + x


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [bs, f, t]
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ConvTemporalGraphical(nn.Module):
    # Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Shape:
        - Input: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Output: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self, time_dim, joints_dim):
        super(ConvTemporalGraphical, self).__init__()

        self.A = nn.Parameter(
            torch.FloatTensor(time_dim, joints_dim, joints_dim)
        )  # learnable, graph-agnostic 3-d adjacency matrix(or edge importance matrix)
        stdv = 1.0 / math.sqrt(self.A.size(1))
        self.A.data.uniform_(-stdv, stdv)

        self.T = nn.Parameter(torch.FloatTensor(joints_dim, time_dim, time_dim))
        stdv = 1.0 / math.sqrt(self.T.size(1))
        self.T.data.uniform_(-stdv, stdv)

        self.joints_dim = joints_dim
        self.time_dim = time_dim

    def forward(self, x):
        x = torch.einsum("nctv,vtq->ncqv", (x, self.T))
        x = torch.einsum("nctv,tvw->nctw", (x, self.A))
        return x.contiguous()


class ConvTemporalGraphicalV1(nn.Module):
    # Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Shape:
        - Input: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Output: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self, time_dim, joints_dim, pose_info):
        super(ConvTemporalGraphicalV1, self).__init__()
        parents = pose_info["parents"]
        TFinger = list(pose_info["TFinger"])
        IFinger = list(pose_info["IFinger"])
        MFinger = list(pose_info["MFinger"])
        RFinger = list(pose_info["RFinger"])
        LFinger = list(pose_info["LFinger"])
        keep_joints = pose_info["keep_joints"]
        dim_use = list(keep_joints)
        self.A = nn.Parameter(
            torch.FloatTensor(time_dim, joints_dim, joints_dim)
        )  # learnable, graph-agnostic 3-d adjacency matrix(or edge importance matrix)
        stdv = 1.0 / math.sqrt(self.A.size(1))
        self.A.data.uniform_(-stdv, stdv)

        self.T = nn.Parameter(torch.FloatTensor(joints_dim, time_dim, time_dim))
        stdv = 1.0 / math.sqrt(self.T.size(1))
        self.T.data.uniform_(-stdv, stdv)

        self.A_s = torch.zeros((1, joints_dim, joints_dim), requires_grad=False, dtype=torch.float)
        for i, dim in enumerate(dim_use):
            self.A_s[0][i][i] = 1
            if parents[dim] in dim_use:
                parent_index = dim_use.index(parents[dim])
                self.A_s[0][i][parent_index] = 1
                self.A_s[0][parent_index][i] = 1
            if dim in TFinger:
                index = TFinger.index(dim)
                index_dim = IFinger[index]
                middle_dim = MFinger[index]
                ring_dim = RFinger[index]
                little_dim = LFinger[index]
                index_index = dim_use.index(index_dim)
                middle_index = dim_use.index(middle_dim)
                ring_index = dim_use.index(ring_dim)
                little_index = dim_use.index(little_dim)
                if index_dim in dim_use:
                    self.A_s[0][i][index_index] = 1
                    self.A_s[0][index_index][i] = 1
                    self.A_s[0][i][middle_index] = 1
                    self.A_s[0][middle_index][i] = 1
                    self.A_s[0][i][ring_index] = 1
                    self.A_s[0][ring_index][i] = 1
                    self.A_s[0][i][little_index] = 1
                    self.A_s[0][little_index][i] = 1

            if dim in IFinger:
                index = IFinger.index(dim)
                thumb_dim = TFinger[index]
                middle_dim = MFinger[index]
                ring_dim = RFinger[index]
                little_dim = LFinger[index]
                thumb_index = dim_use.index(thumb_dim)
                middle_index = dim_use.index(middle_dim)
                ring_index = dim_use.index(ring_dim)
                little_index = dim_use.index(little_dim)
                if thumb_dim in dim_use:
                    self.A_s[0][i][thumb_index] = 1
                    self.A_s[0][thumb_index][i] = 1
                    self.A_s[0][i][middle_index] = 1
                    self.A_s[0][middle_index][i] = 1
                    self.A_s[0][i][ring_index] = 1
                    self.A_s[0][ring_index][i] = 1
                    self.A_s[0][i][little_index] = 1
                    self.A_s[0][little_index][i] = 1

            if dim in MFinger:
                index = MFinger.index(dim)
                index_dim = IFinger[index]
                thumb_dim = TFinger[index]
                ring_dim = RFinger[index]
                little_dim = LFinger[index]
                index_index = dim_use.index(index_dim)
                thumb_index = dim_use.index(thumb_dim)
                ring_index = dim_use.index(ring_dim)
                little_index = dim_use.index(little_dim)
                if index_dim in dim_use:
                    self.A_s[0][i][index_index] = 1
                    self.A_s[0][index_index][i] = 1
                    self.A_s[0][i][thumb_index] = 1
                    self.A_s[0][thumb_index][i] = 1
                    self.A_s[0][i][ring_index] = 1
                    self.A_s[0][ring_index][i] = 1
                    self.A_s[0][i][little_index] = 1
                    self.A_s[0][little_index][i] = 1

            if dim in RFinger:
                index = RFinger.index(dim)
                index_dim = IFinger[index]
                middle_dim = MFinger[index]
                thumb_dim = TFinger[index]
                little_dim = LFinger[index]
                index_index = dim_use.index(index_dim)
                middle_index = dim_use.index(middle_dim)
                thumb_index = dim_use.index(thumb_dim)
                little_index = dim_use.index(little_dim)
                if index_dim in dim_use:
                    self.A_s[0][i][index_index] = 1
                    self.A_s[0][index_index][i] = 1
                    self.A_s[0][i][middle_index] = 1
                    self.A_s[0][middle_index][i] = 1
                    self.A_s[0][i][thumb_index] = 1
                    self.A_s[0][thumb_index][i] = 1
                    self.A_s[0][i][little_index] = 1
                    self.A_s[0][little_index][i] = 1

            if dim in LFinger:
                index = LFinger.index(dim)
                index_dim = IFinger[index]
                middle_dim = MFinger[index]
                ring_dim = RFinger[index]
                thumb_dim = TFinger[index]
                index_index = dim_use.index(index_dim)
                middle_index = dim_use.index(middle_dim)
                ring_index = dim_use.index(ring_dim)
                thumb_index = dim_use.index(thumb_dim)
                if index_dim in dim_use:
                    self.A_s[0][i][index_index] = 1
                    self.A_s[0][index_index][i] = 1
                    self.A_s[0][i][middle_index] = 1
                    self.A_s[0][middle_index][i] = 1
                    self.A_s[0][i][ring_index] = 1
                    self.A_s[0][ring_index][i] = 1
                    self.A_s[0][i][thumb_index] = 1
                    self.A_s[0][thumb_index][i] = 1

        self.joints_dim = joints_dim
        self.time_dim = time_dim

    def forward(self, x):
        A = self.A * self.A_s.to(x.device)
        x = torch.einsum("nctv,vtq->ncqv", (x, self.T))
        x = torch.einsum("nctv,tvw->nctw", (x, A))
        return x.contiguous()


class PromptGenBlock(nn.Module):
    def __init__(self, prompt_dim=16, prompt_len=5, prompt_size=48, lin_dim=384):

        super(PromptGenBlock, self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size))
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x, emb):
        """
        x: bs*nk, feature_dim, dct, joints
        emb: bs*nk, feature_dim
        """
        bs, xyz, dct, joints = x.shape
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)  # [bs*nk,, prompt_len]
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(bs, 1, 1, 1, 1, 1).squeeze(1)

        prompt = torch.sum(prompt, dim=1)
        prompt = F.interpolate(prompt, (dct, joints), mode="bilinear")
        prompt = self.conv3x3(prompt)

        return prompt
