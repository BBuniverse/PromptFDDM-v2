import numpy as np
from motion_pred.utils.dataset import Dataset
from motion_pred.utils.skeleton import Skeleton


class DatasetFPHA(Dataset):

    def __init__(self, mode, t_his=15, t_pred=60, actions="all", **kwargs):
        super().__init__(mode, t_his, t_pred, actions)

    def prepare_data(self):
        self.data_file = "dataset/FPHA_random.p"
        self.subjects_split = {"train": [1, 5, 6], "test": [2, 3, 4]}
        self.subjects = ["Subject_%d" % x for x in self.subjects_split[self.mode]]
        self.skeleton = Skeleton(
            parents=[-1, 0, 0, 0, 0, 0, 1, 6, 7, 2, 9, 10, 3, 12, 13, 4, 15, 16, 5, 18, 19],
            TFinger=[1, 6, 7, 8],
            IFinger=[2, 9, 10, 11],
            MFinger=[3, 12, 13, 14],
            RFinger=[4, 15, 16, 17],
            LFinger=[5, 18, 19, 20],
        )
        self.removed_joints = {}
        self.kept_joints = np.array([x for x in range(21) if x not in self.removed_joints])
        self.skeleton.remove_joints(self.removed_joints)
        self.process_data()

    def process_data(self):
        data_o = np.load(self.data_file, allow_pickle=True)  # [frame, skeleton, xyz] for each subject
        data_f = dict(filter(lambda x: x[0] in self.subjects, data_o.items()))  # S1, S5, S6, S7, S8 for training

        if self.actions != "all":
            for key in list(data_f.keys()):
                data_f[key] = dict(filter(lambda x: all([a in x[0] for a in self.actions]), data_f[key].items()))
                if len(data_f[key]) == 0:
                    data_f.pop(key)
        for data_s in data_f.values():
            for action in data_s.keys():
                seq = data_s[action][:, self.kept_joints, :]  # 3D kept joints movements
                seq[:, 1:] -= seq[:, :1]  # normalize to center point
                data_s[action] = seq  # store the normalized data back
        self.data = data_f
