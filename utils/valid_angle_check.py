import torch
import ipdb


def HO3D_valid_angle_check_torch(p3d):
    """
    p3d: [bs, 20, 3] or [bs, 60]
    """
    if p3d.shape[-1] == 60:
        p3d = p3d.reshape([p3d.shape[0], 20, 3])
    data_all = p3d
    cos_func = lambda p1, p2: torch.sum(p1 * p2, dim=1) / torch.norm(p1, dim=1) / torch.norm(p2, dim=1)
    valid_cos = {}

    # MCP
    p1 = data_all[:, 12]
    p2 = data_all[:, 13] - data_all[:, 12]
    cos_gt = cos_func(p1, p2)
    valid_cos["T_MCP"] = cos_gt

    p1 = data_all[:, 0]
    p2 = data_all[:, 1] - data_all[:, 0]
    cos_gt = cos_func(p1, p2)
    valid_cos["I_MCP"] = cos_gt

    p1 = data_all[:, 3]
    p2 = data_all[:, 4] - data_all[:, 3]
    cos_gt = cos_func(p1, p2)
    valid_cos["M_MCP"] = cos_gt

    p1 = data_all[:, 9]
    p2 = data_all[:, 10] - data_all[:, 9]
    cos_gt = cos_func(p1, p2)
    valid_cos["R_MCP"] = cos_gt

    p1 = data_all[:, 6]
    p2 = data_all[:, 7] - data_all[:, 6]
    cos_gt = cos_func(p1, p2)
    valid_cos["L_MCP"] = cos_gt

    # PIP
    p1 = data_all[:, 14] - data_all[:, 13]
    p2 = data_all[:, 12] - data_all[:, 13]
    cos_gt = cos_func(p1, p2)
    valid_cos["T_PIP"] = cos_gt

    p1 = data_all[:, 0] - data_all[:, 1]
    p2 = data_all[:, 2] - data_all[:, 1]
    cos_gt = cos_func(p1, p2)
    valid_cos["I_PIP"] = cos_gt

    p1 = data_all[:, 3] - data_all[:, 4]
    p2 = data_all[:, 5] - data_all[:, 4]
    cos_gt = cos_func(p1, p2)
    valid_cos["M_PIP"] = cos_gt

    p1 = data_all[:, 9] - data_all[:, 10]
    p2 = data_all[:, 11] - data_all[:, 10]
    cos_gt = cos_func(p1, p2)
    valid_cos["R_PIP"] = cos_gt

    p1 = data_all[:, 6] - data_all[:, 7]
    p2 = data_all[:, 8] - data_all[:, 7]
    cos_gt = cos_func(p1, p2)
    valid_cos["L_PIP"] = cos_gt

    # DIP
    p1 = data_all[:, 15] - data_all[:, 14]
    p2 = data_all[:, 13] - data_all[:, 14]
    cos_gt = cos_func(p1, p2)
    valid_cos["T_DIP"] = cos_gt

    p1 = data_all[:, 16] - data_all[:, 2]
    p2 = data_all[:, 1] - data_all[:, 2]
    cos_gt = cos_func(p1, p2)
    valid_cos["I_DIP"] = cos_gt

    p1 = data_all[:, 17] - data_all[:, 5]
    p2 = data_all[:, 4] - data_all[:, 5]
    cos_gt = cos_func(p1, p2)
    valid_cos["M_DIP"] = cos_gt

    p1 = data_all[:, 18] - data_all[:, 11]
    p2 = data_all[:, 10] - data_all[:, 11]
    cos_gt = cos_func(p1, p2)
    valid_cos["R_DIP"] = cos_gt

    p1 = data_all[:, 19] - data_all[:, 8]
    p2 = data_all[:, 7] - data_all[:, 8]
    cos_gt = cos_func(p1, p2)
    valid_cos["L_DIP"] = cos_gt

    # MCP to MCP
    p1 = data_all[:, 12]  # TFinger
    p2 = data_all[:, 0]
    cos_gt = cos_func(p1, p2)
    valid_cos["MCP_T2I"] = cos_gt
    p2 = data_all[:, 3]
    cos_gt = cos_func(p1, p2)
    valid_cos["MCP_T2M"] = cos_gt
    p2 = data_all[:, 9]
    cos_gt = cos_func(p1, p2)
    valid_cos["MCP_T2R"] = cos_gt
    p2 = data_all[:, 6]
    cos_gt = cos_func(p1, p2)
    valid_cos["MCP_T2L"] = cos_gt

    p1 = data_all[:, 0]  # IFinger
    p2 = data_all[:, 3]
    cos_gt = cos_func(p1, p2)
    valid_cos["MCP_I2M"] = cos_gt
    p2 = data_all[:, 9]
    cos_gt = cos_func(p1, p2)
    valid_cos["MCP_I2R"] = cos_gt
    p2 = data_all[:, 6]
    cos_gt = cos_func(p1, p2)
    valid_cos["MCP_I2L"] = cos_gt

    p1 = data_all[:, 3]  # MFinger
    p2 = data_all[:, 9]
    cos_gt = cos_func(p1, p2)
    valid_cos["MCP_M2R"] = cos_gt
    p2 = data_all[:, 6]
    cos_gt = cos_func(p1, p2)
    valid_cos["MCP_M2L"] = cos_gt

    p1 = data_all[:, 9]  # RFinger
    p2 = data_all[:, 6]
    cos_gt = cos_func(p1, p2)
    valid_cos["MCP_R2L"] = cos_gt

    return valid_cos


def FPHA_valid_angle_check_torch(p3d):
    """
    p3d: [bs, 20, 3] or [bs, 60]
    """
    if p3d.shape[-1] == 60:
        p3d = p3d.reshape([p3d.shape[0], 20, 3])
    data_all = p3d
    cos_func = lambda p1, p2: torch.sum(p1 * p2, dim=1) / torch.norm(p1, dim=1) / torch.norm(p2, dim=1)
    valid_cos = {}

    # MCP
    p1 = data_all[:, 0]
    p2 = data_all[:, 5] - data_all[:, 0]
    cos_gt = cos_func(p1, p2)
    valid_cos["T_MCP"] = cos_gt

    p1 = data_all[:, 1]
    p2 = data_all[:, 8] - data_all[:, 1]
    cos_gt = cos_func(p1, p2)
    valid_cos["I_MCP"] = cos_gt

    p1 = data_all[:, 2]
    p2 = data_all[:, 11] - data_all[:, 2]
    cos_gt = cos_func(p1, p2)
    valid_cos["M_MCP"] = cos_gt

    p1 = data_all[:, 3]
    p2 = data_all[:, 14] - data_all[:, 3]
    cos_gt = cos_func(p1, p2)
    valid_cos["R_MCP"] = cos_gt

    p1 = data_all[:, 4]
    p2 = data_all[:, 17] - data_all[:, 4]
    cos_gt = cos_func(p1, p2)
    valid_cos["L_MCP"] = cos_gt

    # PIP
    p1 = data_all[:, 6] - data_all[:, 5]
    p2 = data_all[:, 0] - data_all[:, 5]
    cos_gt = cos_func(p1, p2)
    valid_cos["T_PIP"] = cos_gt

    p1 = data_all[:, 9] - data_all[:, 8]
    p2 = data_all[:, 1] - data_all[:, 8]
    cos_gt = cos_func(p1, p2)
    valid_cos["I_PIP"] = cos_gt

    p1 = data_all[:, 12] - data_all[:, 11]
    p2 = data_all[:, 2] - data_all[:, 11]
    cos_gt = cos_func(p1, p2)
    valid_cos["M_PIP"] = cos_gt

    p1 = data_all[:, 15] - data_all[:, 14]
    p2 = data_all[:, 3] - data_all[:, 14]
    cos_gt = cos_func(p1, p2)
    valid_cos["R_PIP"] = cos_gt

    p1 = data_all[:, 18] - data_all[:, 17]
    p2 = data_all[:, 4] - data_all[:, 17]
    cos_gt = cos_func(p1, p2)
    valid_cos["L_PIP"] = cos_gt

    # DIP
    p1 = data_all[:, 7] - data_all[:, 6]
    p2 = data_all[:, 5] - data_all[:, 6]
    cos_gt = cos_func(p1, p2)
    valid_cos["T_DIP"] = cos_gt

    p1 = data_all[:, 10] - data_all[:, 9]
    p2 = data_all[:, 8] - data_all[:, 9]
    cos_gt = cos_func(p1, p2)
    valid_cos["I_DIP"] = cos_gt

    p1 = data_all[:, 13] - data_all[:, 12]
    p2 = data_all[:, 11] - data_all[:, 12]
    cos_gt = cos_func(p1, p2)
    valid_cos["M_DIP"] = cos_gt

    p1 = data_all[:, 16] - data_all[:, 15]
    p2 = data_all[:, 14] - data_all[:, 15]
    cos_gt = cos_func(p1, p2)
    valid_cos["R_DIP"] = cos_gt

    p1 = data_all[:, 19] - data_all[:, 18]
    p2 = data_all[:, 17] - data_all[:, 18]
    cos_gt = cos_func(p1, p2)
    valid_cos["L_DIP"] = cos_gt

    # MCP to MCP
    p1 = data_all[:, 0]  # TFinger
    p2 = data_all[:, 1]
    cos_gt = cos_func(p1, p2)
    valid_cos["MCP_T2I"] = cos_gt
    p2 = data_all[:, 2]
    cos_gt = cos_func(p1, p2)
    valid_cos["MCP_T2M"] = cos_gt
    p2 = data_all[:, 3]
    cos_gt = cos_func(p1, p2)
    valid_cos["MCP_T2R"] = cos_gt
    p2 = data_all[:, 4]
    cos_gt = cos_func(p1, p2)
    valid_cos["MCP_T2L"] = cos_gt

    p1 = data_all[:, 1]  # IFinger
    p2 = data_all[:, 2]
    cos_gt = cos_func(p1, p2)
    valid_cos["MCP_I2M"] = cos_gt
    p2 = data_all[:, 3]
    cos_gt = cos_func(p1, p2)
    valid_cos["MCP_I2R"] = cos_gt
    p2 = data_all[:, 4]
    cos_gt = cos_func(p1, p2)
    valid_cos["MCP_I2L"] = cos_gt

    p1 = data_all[:, 2]  # MFinger
    p2 = data_all[:, 3]
    cos_gt = cos_func(p1, p2)
    valid_cos["MCP_M2R"] = cos_gt
    p2 = data_all[:, 4]
    cos_gt = cos_func(p1, p2)
    valid_cos["MCP_M2L"] = cos_gt

    p1 = data_all[:, 3]  # RFinger
    p2 = data_all[:, 4]
    cos_gt = cos_func(p1, p2)
    valid_cos["MCP_R2L"] = cos_gt

    return valid_cos


def H2O_valid_angle_check_torch(p3d):
    """
    p3d: [bs, 20, 3] or [bs, 60]
    """
    if p3d.shape[-1] == 60:
        p3d = p3d.reshape([p3d.shape[0], 20, 3])
    data_all = p3d
    cos_func = lambda p1, p2: torch.sum(p1 * p2, dim=1) / torch.norm(p1, dim=1) / torch.norm(p2, dim=1)
    valid_cos = {}

    # MCP
    p1 = data_all[:, 0]
    p2 = data_all[:, 1] - data_all[:, 0]
    cos_gt = cos_func(p1, p2)
    valid_cos["T_MCP"] = cos_gt

    p1 = data_all[:, 4]
    p2 = data_all[:, 5] - data_all[:, 4]
    cos_gt = cos_func(p1, p2)
    valid_cos["I_MCP"] = cos_gt

    p1 = data_all[:, 8]
    p2 = data_all[:, 9] - data_all[:, 8]
    cos_gt = cos_func(p1, p2)
    valid_cos["M_MCP"] = cos_gt

    p1 = data_all[:, 12]
    p2 = data_all[:, 13] - data_all[:, 12]
    cos_gt = cos_func(p1, p2)
    valid_cos["R_MCP"] = cos_gt

    p1 = data_all[:, 16]
    p2 = data_all[:, 17] - data_all[:, 16]
    cos_gt = cos_func(p1, p2)
    valid_cos["L_MCP"] = cos_gt

    # PIP
    p1 = data_all[:, 2] - data_all[:, 1]
    p2 = data_all[:, 0] - data_all[:, 1]
    cos_gt = cos_func(p1, p2)
    valid_cos["T_PIP"] = cos_gt

    p1 = data_all[:, 6] - data_all[:, 5]
    p2 = data_all[:, 4] - data_all[:, 5]
    cos_gt = cos_func(p1, p2)
    valid_cos["I_PIP"] = cos_gt

    p1 = data_all[:, 10] - data_all[:, 9]
    p2 = data_all[:, 8] - data_all[:, 9]
    cos_gt = cos_func(p1, p2)
    valid_cos["M_PIP"] = cos_gt

    p1 = data_all[:, 14] - data_all[:, 13]
    p2 = data_all[:, 12] - data_all[:, 13]
    cos_gt = cos_func(p1, p2)
    valid_cos["R_PIP"] = cos_gt

    p1 = data_all[:, 18] - data_all[:, 17]
    p2 = data_all[:, 16] - data_all[:, 17]
    cos_gt = cos_func(p1, p2)
    valid_cos["L_PIP"] = cos_gt

    # DIP
    p1 = data_all[:, 3] - data_all[:, 2]
    p2 = data_all[:, 1] - data_all[:, 2]
    cos_gt = cos_func(p1, p2)
    valid_cos["T_DIP"] = cos_gt

    p1 = data_all[:, 7] - data_all[:, 6]
    p2 = data_all[:, 5] - data_all[:, 6]
    cos_gt = cos_func(p1, p2)
    valid_cos["I_DIP"] = cos_gt

    p1 = data_all[:, 11] - data_all[:, 10]
    p2 = data_all[:, 9] - data_all[:, 10]
    cos_gt = cos_func(p1, p2)
    valid_cos["M_DIP"] = cos_gt

    p1 = data_all[:, 15] - data_all[:, 14]
    p2 = data_all[:, 13] - data_all[:, 14]
    cos_gt = cos_func(p1, p2)
    valid_cos["R_DIP"] = cos_gt

    p1 = data_all[:, 19] - data_all[:, 18]
    p2 = data_all[:, 17] - data_all[:, 18]
    cos_gt = cos_func(p1, p2)
    valid_cos["L_DIP"] = cos_gt

    # MCP to MCP
    p1 = data_all[:, 0]  # TFinger
    p2 = data_all[:, 4]
    cos_gt = cos_func(p1, p2)
    valid_cos["MCP_T2I"] = cos_gt
    p2 = data_all[:, 8]
    cos_gt = cos_func(p1, p2)
    valid_cos["MCP_T2M"] = cos_gt
    p2 = data_all[:, 12]
    cos_gt = cos_func(p1, p2)
    valid_cos["MCP_T2R"] = cos_gt
    p2 = data_all[:, 16]
    cos_gt = cos_func(p1, p2)
    valid_cos["MCP_T2L"] = cos_gt

    p1 = data_all[:, 4]  # IFinger
    p2 = data_all[:, 8]
    cos_gt = cos_func(p1, p2)
    valid_cos["MCP_I2M"] = cos_gt
    p2 = data_all[:, 12]
    cos_gt = cos_func(p1, p2)
    valid_cos["MCP_I2R"] = cos_gt
    p2 = data_all[:, 16]
    cos_gt = cos_func(p1, p2)
    valid_cos["MCP_I2L"] = cos_gt

    p1 = data_all[:, 8]  # MFinger
    p2 = data_all[:, 12]
    cos_gt = cos_func(p1, p2)
    valid_cos["MCP_M2R"] = cos_gt
    p2 = data_all[:, 16]
    cos_gt = cos_func(p1, p2)
    valid_cos["MCP_M2L"] = cos_gt

    p1 = data_all[:, 12]  # RFinger
    p2 = data_all[:, 16]
    cos_gt = cos_func(p1, p2)
    valid_cos["MCP_R2L"] = cos_gt

    return valid_cos
