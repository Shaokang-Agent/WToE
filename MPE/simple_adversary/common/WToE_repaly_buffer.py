import threading
import numpy as np


class WToE_Buffer:
    def __init__(self, args):
        self.size = args.buffer_size
        self.args = args
        # memory management
        self.current_size = 0
        self.current_index = -1
        # create the buffer to store info
        self.buffer = dict()
        for i in range(self.args.n_agents):
            self.buffer['o_%d' % i] = np.empty([self.size, self.args.obs_shape[i]])
            self.buffer['u_%d' % i] = np.empty([self.size, self.args.action_shape[i]])
            self.buffer['r_%d' % i] = np.empty([self.size])
            self.buffer['o_next_%d' % i] = np.empty([self.size, self.args.obs_shape[i]])
        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, o, u, r, o_next):
        self._get_storage_idx()  # 以transition的形式存，每次只存一条经验
        for i in range(self.args.n_agents):
            with self.lock:
                self.buffer['o_%d' % i][self.current_index] = o[i]
                self.buffer['u_%d' % i][self.current_index] = u[i]
                self.buffer['r_%d' % i][self.current_index] = r[i]
                self.buffer['o_next_%d' % i][self.current_index] = o_next[i]

    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffer = {}
        idxs = []
        while(1):
            idx = np.random.randint(0, self.current_size)
            if self.current_size < self.size and idx - self.args.history_size < 0:
                continue
            idxs.append(idx)
            if len(idxs) >= self.args.batch_size:
                break
        idxs = np.array(idxs)
        for key in self.buffer.keys():
            temp_buffer[key] = self.buffer[key][idxs]
            temp_buffer[key] = np.expand_dims(temp_buffer[key], axis=0)
            for num in range(self.args.history_size-1):
                new_idxs = idxs-num-1
                temp_buffer[key] = np.insert(temp_buffer[key], 0, self.buffer[key][new_idxs], 0)
            if "u" in key:
                temp_buffer[key] = np.insert(temp_buffer[key], 0, self.buffer[key][idxs-self.args.history_size], 0)
        return temp_buffer

    def _get_storage_idx(self):
        self.current_index = (self.current_index + 1) % self.size
        self.current_size = np.min([self.current_size+1, self.size])

