import numpy as np


class Skeleton:
    def __init__(self, parents, TFinger, IFinger, MFinger, RFinger, LFinger):
        assert len(TFinger) == len(IFinger) == len(MFinger) == len(RFinger) == len(LFinger)

        self._parents = np.array(parents)
        self._TFinger = TFinger
        self._IFinger = IFinger
        self._MFinger = MFinger
        self._RFinger = RFinger
        self._LFinger = LFinger
        self._compute_metadata()

    def num_joints(self):
        return len(self._parents)

    def parents(self):
        return self._parents

    def has_children(self):
        return self._has_children

    def children(self):
        return self._children

    def remove_joints(self, joints_to_remove):
        """
        Remove the joints specified in 'joints_to_remove'.
        """
        valid_joints = []
        for joint in range(len(self._parents)):
            if joint not in joints_to_remove:
                valid_joints.append(joint)

        for i in range(len(self._parents)):
            while self._parents[i] in joints_to_remove:
                self._parents[i] = self._parents[self._parents[i]]

        index_offsets = np.zeros(len(self._parents), dtype=int)
        new_parents = []
        for i, parent in enumerate(self._parents):
            if i not in joints_to_remove:
                new_parents.append(parent - index_offsets[parent])
            else:
                index_offsets[i:] += 1
        self._parents = np.array(new_parents)

        if self._TFinger is not None:
            new_TFinger = []
            for joint in self._TFinger:
                if joint in valid_joints:
                    new_TFinger.append(joint - index_offsets[joint])
            self._TFinger = new_TFinger
        if self._IFinger is not None:
            new_IFinger = []
            for joint in self._IFinger:
                if joint in valid_joints:
                    new_IFinger.append(joint - index_offsets[joint])
            self._IFinger = new_IFinger
        if self._MFinger is not None:
            new_MFinger = []
            for joint in self._MFinger:
                if joint in valid_joints:
                    new_MFinger.append(joint - index_offsets[joint])
            self._MFinger = new_MFinger
        if self._RFinger is not None:
            new_RFinger = []
            for joint in self._RFinger:
                if joint in valid_joints:
                    new_RFinger.append(joint - index_offsets[joint])
            self._RFinger = new_RFinger
        if self._LFinger is not None:
            new_LFinger = []
            for joint in self._LFinger:
                if joint in valid_joints:
                    new_LFinger.append(joint - index_offsets[joint])
            self._LFinger = new_LFinger

        self._compute_metadata()

        return valid_joints

    def TFinger(self):
        return self._TFinger

    def IFinger(self):
        return self._IFinger

    def MFinger(self):
        return self._MFinger

    def RFinger(self):
        return self._RFinger

    def LFinger(self):
        return self._LFinger

    def _compute_metadata(self):
        self._has_children = np.zeros(len(self._parents)).astype(bool)
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True

        self._children = []
        for i, parent in enumerate(self._parents):
            self._children.append([])
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._children[parent].append(i)
