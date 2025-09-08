from models import LinNF, STEN
from models.diffusion import Diffusion
from models.transformer import MotionTransformer
from utils.util import get_dct_matrix
import ipdb


def get_model(cfg, args, dataset):
    traj_dim = dataset.traj_dim // 3
    model_type = cfg.dataset
    if "nf" not in cfg.id:
        # drop the root joint
        keep_joints = dataset.kept_joints[1:]
        if model_type in ["HO3D"]:
            parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14, 15, 3, 6, 12, 9]
            TFinger = [13, 14, 15, 16]
            IFinger = [1, 2, 3, 17]
            MFinger = [4, 5, 6, 18]
            RFinger = [10, 11, 12, 19]
            LFinger = [7, 8, 9, 20]
        elif model_type == "FPHA":
            parents = [-1, 0, 0, 0, 0, 0, 1, 6, 7, 2, 9, 10, 3, 12, 13, 4, 15, 16, 5, 18, 19]
            TFinger = [1, 6, 7, 8]
            IFinger = [2, 9, 10, 11]
            MFinger = [3, 12, 13, 14]
            RFinger = [4, 15, 16, 17]
            LFinger = [5, 18, 19, 20]
        elif model_type in ["H2O"]:
            parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
            TFinger = [1, 2, 3, 4]
            IFinger = [5, 6, 7, 8]
            MFinger = [9, 10, 11, 12]
            RFinger = [13, 14, 15, 16]
            LFinger = [17, 18, 19, 20]

        pose_info = {
            "keep_joints": keep_joints,
            "parents": parents,
            "TFinger": TFinger,
            "IFinger": IFinger,
            "MFinger": MFinger,
            "RFinger": RFinger,
            "LFinger": LFinger,
        }
        return (STEN.STEN(cfg=cfg, args=args, input_channels=3, joints_to_consider=traj_dim, pose_info=pose_info).to(cfg.device)), LinNF.LinNF(
            data_dim=traj_dim * 3, num_layer=cfg.STEN_cfg["specs"]["num_flow_layer"]
        ).to(cfg.device)

    elif "nf" in cfg.id:
        return LinNF.LinNF(data_dim=dataset.traj_dim, num_layer=cfg.STEN_cfg["specs"]["num_flow_layer"])


def get_diffusion(cfg, args):
    # RDGN cfg
    dct_m, idct_m = get_dct_matrix(cfg.STEN_cfg["t_his"] + cfg.STEN_cfg["t_pred"])
    denoise = MotionTransformer(cfg=cfg, prompt=args.promptFlag, args=args, dct_m=dct_m)
    diffusion = Diffusion(cfg=cfg.RDGN_cfg, dct_m=dct_m, idct_m=idct_m)
    return diffusion, denoise
