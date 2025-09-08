import numpy as np
from motion_pred.utils.dataset import Dataset
from motion_pred.utils.skeleton import Skeleton


class DatasetHO3D(Dataset):

    def __init__(self, mode, t_his=25, t_pred=100, actions="all", use_vel=False):
        self.use_vel = use_vel
        super().__init__(mode, t_his, t_pred, actions)
        if use_vel:
            self.traj_dim += 3

    def prepare_data(self):
        self.data_file = "dataset/HO3D_v3_random.p"
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

        if self.actions != "all":
            for key in list(data_f.keys()):
                data_f[key] = dict(filter(lambda x: any([a in str.lower(x[0]) for a in self.actions]), data_f[key].items()))
                if len(data_f[key]) == 0:
                    data_f.pop(key)
        for data_s in data_f.values():
            for action in data_s.keys():
                seq = data_s[action]
                if self.use_vel:
                    v = (np.diff(seq[:, :1], axis=0) * 50).clip(-5.0, 5.0)
                    v = np.append(v, v[[-1]], axis=0)
                seq[:, 1:] -= seq[:, :1]
                if self.use_vel:
                    seq = np.concatenate((seq, v), axis=1)
                data_s[action] = seq
        self.data = data_f
