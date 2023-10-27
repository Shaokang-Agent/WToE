import numpy as np
import torch
from algorithm.WToE import WToE


class WToE_Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        self.policy = WToE(args, agent_id)

    def select_action(self, o, noise_rate, epsilon=0, steps=0, s_stack=[], a_stack=[], r_stack=[]):
        s_stack = np.array(s_stack)
        a_stack = np.array(a_stack)
        r_stack = np.array(r_stack)
        if np.random.uniform() < epsilon:
            u = np.random.uniform(-self.args.high_action, self.args.high_action, self.args.action_shape[self.agent_id])
        else:
            if a_stack.shape[0] == 0:
                inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
                if self.args.cuda:
                    inputs = inputs.cuda()
                pi = self.policy.actor_network(inputs).squeeze(0)

                #pi_pred = self.policy.vae.action_decoder(embedding, obs, prev_actions, next_obs, reward)
                # print('{} : {}'.format(self.name, pi))
                u = pi.cpu().numpy()
                noise = noise_rate * self.args.high_action * np.random.randn(*u.shape)  # gaussian noise
                u += noise
                u = np.clip(u, -self.args.high_action, self.args.high_action)
            else:
                inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
                if self.args.cuda:
                    inputs = inputs.cuda()
                pi = self.policy.actor_network(inputs).squeeze(0)
                s_t_1 = torch.from_numpy(s_stack[1:self.args.history_size+1]).float().unsqueeze(dim=1)
                s_t = torch.from_numpy(s_stack[2:self.args.history_size+2]).float().unsqueeze(dim=1)
                a_t_2 = torch.from_numpy(a_stack[0:self.args.history_size]).float().unsqueeze(dim=1)
                a_t_1 = torch.from_numpy(a_stack[1:self.args.history_size+1]).float().unsqueeze(dim=1)
                r_t_1 = torch.from_numpy(r_stack[1:self.args.history_size+1]).float().unsqueeze(dim=1).unsqueeze(dim=1)

                if self.args.cuda:
                    s_t_1 = s_t_1.cuda()
                    s_t = s_t.cuda()
                    a_t_2 = a_t_2.cuda()
                    a_t_1 = a_t_1.cuda()
                    r_t_1 = r_t_1.cuda()
                _, pi_pred = self.policy.vae.compute_vae_loss(s_t_1, a_t_2, a_t_1, r_t_1, s_t)
                u_pred = pi_pred[-1,-1,:].cpu().numpy()
                u_actor = self.policy.actor_network(s_t[-2,...]).squeeze(0).cpu().numpy()
                u = pi.cpu().numpy()
                bvae_rate = np.max([self.args.bvae_rate_final, self.args.bvae_rate_inital+(self.args.bvae_rate_final-self.args.bvae_rate_inital)*steps*2/self.args.time_steps])
                bvae_noise = bvae_rate * np.abs(u_pred - u_actor) * np.random.randn(*u.shape)
                # if steps % 500 == 0:
                #     print(u, bvae_noise)
                u += bvae_noise
                u = np.clip(u, -self.args.high_action, self.args.high_action)
        return u.copy()

    def learn(self, transitions, other_agents):
        self.policy.train(transitions, other_agents)
