import numpy as np
from model.vae import VAE
import torch
class WToE_Q_learning():
    """docstring for DQN"""
    def __init__(self, args, state_num, action_num):
        super(WToE_Q_learning, self).__init__()
        self.action_num = action_num
        self.Q_value = np.zeros([state_num, action_num])
        self.vae = VAE(args)
        self.learn_step = 0

    def choose_action(self, state, episolon, count, s_stack, a_stack, r_stack):
        Flag = 0
        action_value = self.Q_value[state]
        max_action = np.where(action_value == np.max(action_value))
        action = max_action[0][np.random.randint(0, len(max_action))]
        z_norm = 0
        if count > 0:
            s_t = torch.from_numpy(np.array(s_stack)[2:]).to(torch.int64).unsqueeze(dim=0).unsqueeze(dim=2)
            a_t_1 = torch.from_numpy(np.array(a_stack)[1:-1]).to(torch.int64).unsqueeze(dim=0).unsqueeze(dim=2)
            a_t = torch.from_numpy(np.array(a_stack)[2:]).to(torch.int64).unsqueeze(dim=0).unsqueeze(dim=2)
            r_t = torch.from_numpy(np.array(r_stack)[2:]).float().unsqueeze(dim=0).unsqueeze(dim=2)
            _, action_pred, z = self.vae.compute_vae_loss(s_t, a_t_1, a_t, r_t)
            z_norm = torch.norm(z)
            action_value_1 = self.Q_value[state]
            max_action_1 = np.where(action_value_1 == np.max(action_value_1))
            action_1 = max_action_1[0][np.random.randint(0, len(max_action_1))]
            if np.argmax(action_pred.detach()[0, -1].numpy()) != action_1:
                Flag = 1
                if np.random.rand() >= 1 - episolon + 0.7:
                    action = np.argmax(action_pred.detach()[0, -1].numpy())
        if np.random.rand() >= episolon:
            action = np.random.randint(0, self.action_num)
        return action, Flag, z_norm

    def learn(self, state, action, reward, next_state, replay_buffer):

        self.Q_value[state, action] = reward + 0.99 * np.max(self.Q_value[next_state])

        if self.learn_step % 20 == 0:
            s_tr, a_tr, r_tr, s_n_tr = replay_buffer.sample()
            s_trajectory = torch.from_numpy(np.array(s_tr)).to(torch.int64)
            a_prev_trajectory = torch.from_numpy(np.array(a_tr)[:,:-1,:]).to(torch.int64)
            a_trajectory = torch.from_numpy(np.array(a_tr)[:, 1:, :]).to(torch.int64)
            r_trajectory = torch.from_numpy(np.array(r_tr)).float()

            vae_loss, _ , _= self.vae.compute_vae_loss(s_trajectory, a_prev_trajectory,
                                                    a_trajectory, r_trajectory)

            self.vae.optimiser_vae.zero_grad()
            vae_loss.backward()
            self.vae.optimiser_vae.step()

        self.learn_step+=1

    def get_Q_value(self):
        return self.Q_value

    def set_Q_value(self, Q):
        self.Q_value = Q
