import os
import yaml
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import pdist, squareform
from motion_pred.utils.dataset_HO3D_multimodal import DatasetHO3D
from motion_pred.utils.dataset_FPHA_multimodal import DatasetFPHA
from motion_pred.utils.dataset_H2O_multimodal import DatasetH2O

import ipdb


class Config:
    def __init__(self, cfg_id):
        self.id = cfg_id
        cfg_name = "motion_pred/cfg/%s.yml" % cfg_id
        if not os.path.exists(cfg_name):
            print("Config file doesn't exist: %s" % cfg_name)
            exit(0)

        cfg = yaml.safe_load(open(cfg_name, "r"))
        self.cfg = cfg
        self.STEN_cfg = cfg["STEN_cfg"]
        self.RDGN_cfg = cfg["RDGN_cfg"]

        # Config root dir
        self.base_dir = "result"

        self.cfg_dir = f"{self.base_dir}/{self.cfg['name']}"
        self.models = f"{self.cfg_dir}/models"
        self.visualization = f"{self.cfg_dir}/visualization"
        self.log_dir = f"{self.cfg_dir}"
        self.tb_dir = f"{self.cfg_dir}/tb"
        self.STEN_path = os.path.join(self.models, "STEN/STEN_%04d.p")
        self.RDGN_path = os.path.join(self.models, "RDGN/RDGN_%s_%04d_STEN_%04d.p")
        self.nf_path = os.path.join(self.models, "vae_%04d.p")
        os.makedirs(self.models, exist_ok=True)
        os.makedirs(os.path.join(self.models, "STEN"), exist_ok=True)
        os.makedirs(os.path.join(self.models, "RDGN"), exist_ok=True)
        os.makedirs(self.visualization, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)

        self.RDGN = cfg.get("RDGN", False)
        self.RDGN_cfg["n_pre"] = self.STEN_cfg["specs"]["n_pre"]
        self.RDGN_cfg["features"] = self.RDGN_cfg.get("features", 320)  # Config DM latent dim
        self.dataset = cfg.get("dataset")
        self.normalize_data = cfg.get("normalize_data", False)
        self.multimodal_threshold = cfg.get("multimodal_threshold", 0.2)
        self.scale = cfg.get("scale", 9)


def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = "\n"
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += " " * (indent_level * 2) + k + ":["
            msg += dict2str(v, indent_level + 1)
            msg += " " * (indent_level * 2) + "]\n"
        else:
            msg += " " * (indent_level * 2) + k + ": " + str(v) + "\n"
    return msg


def get_multimodal_gt(dataset_test, cfg, logger):
    all_data = []
    data_cfg = cfg.STEN_cfg
    data_gen = dataset_test.iter_generator(step=data_cfg["t_his"])

    for data, _, _, _ in tqdm(data_gen):

        data = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
        all_data.append(data)

    all_data = np.concatenate(all_data, axis=0)
    all_start_pose = all_data[:, data_cfg["t_his"] - 1, :]
    pd = squareform(pdist(all_start_pose))
    traj_gt_arr = []
    num_mult = []
    for i in range(pd.shape[0]):
        ind = np.nonzero(pd[i] < cfg.multimodal_threshold)
        traj_gt_arr.append(all_data[ind][:, data_cfg["t_his"] :, :])
        num_mult.append(len(ind[0]))
    logger.info("=" * 40 + " Start " + "=" * 40)
    num_mult = np.array(num_mult)
    logger.info(f"#1 future: {len(np.where(num_mult == 1)[0])}/{pd.shape[0]}")
    logger.info(f"#<10 future: {len(np.where(num_mult < 10)[0])}/{pd.shape[0]}")
    return traj_gt_arr


def dataset_split(cfg, logger):
    """
    output: dataset_dict, dataset_multi_test
    dataset_dict has two keys: 'train', 'test' for enumeration in train and validation.
    dataset_multi_test is used to create multi-modal data for metrics.
    """
    STEN_cfg = cfg.STEN_cfg
    dataset_cls = globals()[f"Dataset{cfg.dataset}"]  # dynamically load the dataset
    dataset = dataset_cls(
        "train",
        STEN_cfg["t_his"],
        STEN_cfg["t_pred"],
        actions="all",
        multimodal_path=STEN_cfg["specs"]["multimodal_path"],
        data_candi_path=STEN_cfg["specs"]["data_candi_path"],
    )
    dataset_test = dataset_cls(
        "test",
        STEN_cfg["t_his"],
        STEN_cfg["t_pred"],
        actions="all",
        multimodal_path=STEN_cfg["specs"]["multimodal_path"],
        data_candi_path=STEN_cfg["specs"]["data_candi_path"],
    )
    if cfg.normalize_data:
        dataset.normalize_data()
        dataset_test.normalize_data(dataset.mean, dataset.std)

    traj_gt_arr = get_multimodal_gt(dataset_test, cfg, logger)

    return {"train": dataset, "test": dataset_test}, traj_gt_arr


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative pathes.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith(".") and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = os.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)
