import torch
import os
from model.actor_critic import Actor, Critic
import numpy as np

class MADDPG:
    def __init__(self, args, agent_id):
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0

        # create the network
        self.actor_network = Actor(args, agent_id)
        self.critic_network = Critic(args)

        # build up the target network
        self.actor_target_network = Actor(args, agent_id)
        self.critic_target_network = Critic(args)

        if args.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())


        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    # update the network
    def train(self, transitions, other_agents):
        r = transitions['r_%d' % self.agent_id]
        o, u, o_next = [], [], []
        for agent_id in range(self.args.n_agents):
            o.append(transitions['o_%d' % agent_id])
            u.append(transitions['u_%d' % agent_id])
            o_next.append(transitions['o_next_%d' % agent_id])

        r = torch.from_numpy(np.array(r)).float()
        o_0 = torch.from_numpy(np.array(o[0])).float()
        o_next_0 = torch.from_numpy(np.array(o_next[0])).float()

        if self.args.n_agents == 2:
            o_1 = torch.from_numpy(np.array(o[1])).float()
            o_next_1 = torch.from_numpy(np.array(o_next[1])).float()
            s = torch.cat([torch.from_numpy(np.array(o[0])).float(), torch.from_numpy(np.array(o[1])).float()], dim=1)
            s_next = torch.cat([torch.from_numpy(np.array(o_next[0])).float(), torch.from_numpy(np.array(o_next[1])).float()] , dim=1)
        else:
            s = o_0
            s_next = o_next_0
        u = torch.from_numpy(np.array(u)).float()
        if self.args.cuda:
            r = r.cuda()
            s = s.cuda()
            u = u.cuda()
            s_next = s_next.cuda()
            o_0 = o_0.cuda()
            o_next_0 = o_next_0.cuda()
            if self.args.n_agents == 2:
                o_1 = o_1.cuda()
                o_next_1 = o_next_1.cuda()
        if self.args.n_agents == 2:
            o = [o_0, o_1]
            o_next = [o_next_0, o_next_1]
        else:
            o = [o_0]
            o_next = [o_next_0]
        # calculate the target Q value function
        u_next = []
        with torch.no_grad():
            index = 0
            for agent_id in range(self.args.n_agents):
                if agent_id == self.agent_id:
                    u_next.append(self.actor_target_network(o[agent_id]))
                else:
                    u_next.append(other_agents[index].policy.actor_target_network(o_next[agent_id]))
                    index += 1
            u_next = torch.stack(u_next)
            if self.args.cuda:
                u_next = u_next.cuda()
            q_next = self.critic_target_network(s_next, u_next).detach()

            target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach()

        # the q loss
        q_value = self.critic_network(s, u)
        critic_loss = (target_q - q_value).pow(2).mean()

        # the actor loss
        u[self.agent_id] = self.actor_network(o[self.agent_id])
        actor_loss = - self.critic_network(s, u).mean()

        if self.train_step % 2 == 0:
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

        self._soft_update_target_network()
        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)
        self.train_step += 1

    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        model_path = os.path.join(model_path, self.args.algorithm)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.actor_network.state_dict(), model_path + '/' + num + '_actor_params.pkl')
        torch.save(self.critic_network.state_dict(),  model_path + '/' + num + '_critic_params.pkl')


