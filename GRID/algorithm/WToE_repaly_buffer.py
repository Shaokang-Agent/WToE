import numpy as np


class WToE_Buffer:
    def __init__(self, args):
        self.size = args.buffer_size
        self.args = args
        # memory management
        self.current_size = 0
        self.current_index = -1
        # create the buffer to store info
        self.s = np.empty([self.size, 1])
        self.a = np.empty([self.size, 1])
        self.r = np.empty([self.size, 1])
        self.s_ = np.empty([self.size, 1])

    # store the episode
    def store_episode(self, s, a, r, s_next):
        self._get_storage_idx()  # 以transition的形式存，每次只存一条经验
        self.s[self.current_index] = s
        self.a[self.current_index] = a
        self.r[self.current_index] = r
        self.s_[self.current_index] = s_next

    # sample the data from the replay buffer
    def sample(self):
        idxs = []
        while(1):
            idx = np.random.randint(0, self.current_size)
            if idx - self.args.history_size < 0:
                continue
            idxs.append(idx)
            if len(idxs) >= self.args.batch_size:
                break
        idxs = np.array(idxs)
        s_trajetory = []
        a_trajectory = []
        r_trajectory = []
        s_next_trajectory = []
        for i in idxs:
            s_trajetory.append(self.s[i-self.args.history_size+1:i+1])
            a_trajectory.append(self.a[i-self.args.history_size:i+1])
            r_trajectory.append(self.r[i - self.args.history_size + 1:i + 1])
            s_next_trajectory.append(self.s_[i - self.args.history_size + 1:i + 1])
        return s_trajetory, a_trajectory, r_trajectory, s_next_trajectory

    def _get_storage_idx(self):
        self.current_index = (self.current_index + 1) % self.size
        self.current_size = np.min([self.current_size+1, self.size])

