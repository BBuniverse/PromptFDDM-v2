import numpy as np
from motion_pred.utils.dataset import Dataset
from motion_pred.utils.skeleton import Skeleton
from utils import util
import torch
import ipdb


class DatasetHO3D(Dataset):

    def __init__(self, mode, t_his=20, t_pred=80, actions="all", **kwargs):
        self.multimodal_path = kwargs["multimodal_path"] if "multimodal_path" in kwargs.keys() else None
        self.data_candi_path = kwargs["data_candi_path"] if "data_candi_path" in kwargs.keys() else None

        super().__init__(mode, t_his, t_pred, actions)

    def prepare_data(self):
        self.data_file = "dataset/HO3D_v3_random.p"
        self.history_caption_all = torch.load("models/text_model/HO3D_caption_simple_en_tensor.p", weights_only=True)
        self.subjects_split = {"train": [], "test": []}
        self.subjects = [x for x in self.subjects_split[self.mode]]
        self.skeleton = Skeleton(
            parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14, 15, 3, 6, 12, 9],
            TFinger=[13, 14, 15, 16],
            IFinger=[1, 2, 3, 17],
            MFinger=[4, 5, 6, 18],
            RFinger=[10, 11, 12, 19],
            LFinger=[7, 8, 9, 20],
        )
        self.removed_joints = {}
        self.kept_joints = np.array([x for x in range(21) if x not in self.removed_joints])
        self.skeleton.remove_joints(self.removed_joints)
        self.process_data()

    def process_data(self):
        data_o = np.load(self.data_file, allow_pickle=True)  # [frame, skeleton, xyz] for each subject
        data_f = data_o[self.mode]

        self.subjects_split["train"] = list(data_o["train"].keys())
        self.subjects_split["test"] = list(data_o["test"].keys())
        self.subjects = [x for x in self.subjects_split[self.mode]]
        self.history_caption = self.history_caption_all[self.mode]

        if self.multimodal_path is None:
            self.data_multimodal = np.load(
                "dataset/HO3D_multi_modal/t_his20_1_thre0.050_t_pred80_thre0.100_index_subAll_random.npz",
                allow_pickle=True,
            )["data_multimodal"].item()
            data_candi = np.load(
                "dataset/HO3D_multi_modal/data_multimodal_t_his20_t_pred80_skiprate15_random.npz",
                allow_pickle=True,
            )["data_candidate.npy"]
        else:
            self.data_multimodal = np.load(self.multimodal_path, allow_pickle=True)["data_multimodal"].item()
            data_candi = np.load(self.data_candi_path, allow_pickle=True)["data_candidate.npy"]

        self.data_candi = {}

        for key in list(data_f.keys()):
            data_f[key] = dict(
                filter(
                    lambda x: (self.actions == "all" or any([a in x[0] for a in self.actions])) and x[1].shape[0] >= self.t_total,
                    data_f[key].items(),
                )
            )
            if len(data_f[key]) == 0:
                data_f.pop(key)
        for sub in data_f.keys():
            data_s = data_f[sub]
            for action in data_s.keys():
                seq = data_s[action][:, self.kept_joints, :]
                seq[:, 1:] -= seq[:, :1]
                data_s[action] = seq

                if sub not in self.data_candi.keys():
                    x0 = np.copy(seq[None, :1, ...])
                    x0[:, :, 0] = 0
                    self.data_candi[sub] = util.absolute2relative(data_candi, parents=self.skeleton.parents(), invert=True, x0=x0)
        self.data = data_f

    def sample(self, n_modality=10):
        subject = np.random.choice(self.subjects)
        dict_s = self.data[subject]
        action = np.random.choice(list(dict_s.keys()))
        seq = dict_s[action]
        fr_start = np.random.randint(seq.shape[0] - self.t_total)
        fr_end = fr_start + self.t_total
        traj = seq[fr_start:fr_end]

        # # ==================== Paper visualization only ====================
        # gitList = [
        #     "SMu1_mug 0_20",
        # ]

        # temp = np.random.choice(gitList)
        # subject, action, fr_start = temp.split("_")
        # fr_start = int(fr_start)
        # seq = self.data[subject][action]
        # fr_end = fr_start + self.t_total
        # traj = seq[fr_start:fr_end]
        # # ==================== Paper visulization only ====================

        # ==================== for Video caption ====================
        history_caption = self.history_caption[subject][action]
        assert seq.shape[0] - self.t_total == len(history_caption)
        caption = history_caption[fr_start]
        # ==================== for Video caption ====================

        if n_modality > 0 and subject in self.data_multimodal.keys():
            candi_tmp = self.data_candi[subject]

            idx_multi = self.data_multimodal[subject][action][fr_start]
            traj_multi = candi_tmp[idx_multi]

            if len(traj_multi) > 0:
                traj_multi[:, : self.t_his] = traj[None, ...][:, : self.t_his]
                if traj_multi.shape[0] > n_modality:
                    st0 = np.random.get_state()
                    idxtmp = np.random.choice(np.arange(traj_multi.shape[0]), n_modality, replace=False)
                    traj_multi = traj_multi[idxtmp]
                    np.random.set_state(st0)
            traj_multi = np.concatenate([traj_multi, np.zeros_like(traj[None, ...][[0] * (n_modality - traj_multi.shape[0])])], axis=0)

            return traj[None, ...], traj_multi, subject + "_" + action, caption, fr_start
        else:
            return traj[None, ...], None, subject + "_" + action, caption, fr_start

    def sampling_generator(self, num_samples=2000, batch_size=16, n_modality=10):
        for i in range(num_samples // batch_size):
            sample = []
            sample_multi = []
            actions = []
            captions = []
            for i in range(batch_size):
                sample_i, sample_multi_i, _, caption, _ = self.sample(n_modality=n_modality)
                sample.append(sample_i)
                sample_multi.append(sample_multi_i[None, ...])
                captions.append(caption)
            sample = np.concatenate(sample, axis=0)
            sample_multi = np.concatenate(sample_multi, axis=0)
            yield sample, sample_multi, actions, captions

    def iter_generator(self, step=25, n_modality=10):
        for sub in self.data.keys():
            data_s = self.data[sub]
            candi_tmp = self.data_candi[sub]
            for act in data_s.keys():
                seq = data_s[act]
                seq_len = seq.shape[0]
                for i in range(0, seq_len - self.t_total, step):
                    traj = seq[None, i : i + self.t_total]
                    caption = self.history_caption[sub][act][i]
                    if n_modality > 0:
                        margin_f = 1
                        thre_his = 0.05
                        thre_pred = 0.1
                        x0 = np.copy(traj)
                        x0[:, :, 0] = 0
                        # observation distance
                        dist_his = np.mean(
                            np.linalg.norm(
                                x0[:, self.t_his - margin_f : self.t_his, 1:] - candi_tmp[:, self.t_his - margin_f : self.t_his, 1:],
                                axis=3,
                            ),
                            axis=(1, 2),
                        )
                        idx_his = np.where(dist_his <= thre_his)[0]

                        # future distance
                        dist_pred = np.mean(
                            np.linalg.norm(x0[:, self.t_his :, 1:] - candi_tmp[idx_his, self.t_his :, 1:], axis=3),
                            axis=(1, 2),
                        )

                        idx_pred = np.where(dist_pred >= thre_pred)[0]
                        traj_multi = candi_tmp[idx_his[idx_pred]]
                        if len(traj_multi) > 0:
                            traj_multi[:, : self.t_his] = traj[:, : self.t_his]
                            if traj_multi.shape[0] > n_modality:
                                traj_multi = traj_multi[:n_modality]
                        traj_multi = np.concatenate([traj_multi, np.zeros_like(traj[[0] * (n_modality - traj_multi.shape[0])])], axis=0)
                    else:
                        traj_multi = None

                    yield traj, traj_multi, act, caption
