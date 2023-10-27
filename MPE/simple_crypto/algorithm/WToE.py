import torch
import os
from model.actor_critic import Actor, Critic
from model.vae import VAE
import numpy as np

class WToE:
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

        self.vae = VAE(args, agent_id)
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
        r_trajectiory = transitions['r_%d' % self.agent_id]
        o_trajectiory, u_trajectiory, o_next_trajectiory = [], [], []
        for agent_id in range(self.args.n_agents):
            o_trajectiory.append(transitions['o_%d' % agent_id])
            u_trajectiory.append(transitions['u_%d' % agent_id])
            o_next_trajectiory.append(transitions['o_next_%d' % agent_id])

        r_trajectiory = torch.from_numpy(np.array(r_trajectiory)).float()
        u_trajectiory = torch.from_numpy(np.array(u_trajectiory)).float()
        u_prev_trajectiory = u_trajectiory[:,:-1,...]
        u_trajectiory = u_trajectiory[:,1:,...]

        for agent_id in range(self.args.n_agents):
            o_trajectiory[agent_id] = torch.from_numpy(np.array(o_trajectiory[agent_id])).float()
            o_next_trajectiory[agent_id] = torch.from_numpy(np.array(o_next_trajectiory[agent_id])).float()
        s_trajectiory = torch.cat(o_trajectiory, dim=2)
        s_next_trajectiory = torch.cat(o_next_trajectiory, dim=2)
        if self.args.cuda:
            r_trajectiory = r_trajectiory.cuda()
            s_trajectiory = s_trajectiory.cuda()
            u_trajectiory = u_trajectiory.cuda()
            u_prev_trajectiory = u_prev_trajectiory.cuda()
            s_next_trajectiory = s_next_trajectiory.cuda()
            for agent_id in range(self.args.n_agents):
                o_trajectiory[agent_id] = o_trajectiory[agent_id].cuda()
                o_next_trajectiory[agent_id] = o_next_trajectiory[agent_id].cuda()

        # calculate the target Q value function
        s = s_trajectiory[-1]
        u = u_trajectiory[:,-1,...]
        r = r_trajectiory[-1]
        s_next = s_next_trajectiory[-1]
        u_next = []
        with torch.no_grad():
            index = 0
            for agent_id in range(self.args.n_agents):
                if agent_id == self.agent_id:
                    u_next.append(self.actor_target_network(o_next_trajectiory[agent_id][-1]))
                else:
                    u_next.append(other_agents[index].policy.actor_target_network(o_next_trajectiory[agent_id][-1]))
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
        u[self.agent_id] = self.actor_network(o_trajectiory[self.agent_id][-1])
        actor_loss = - self.critic_network(s, u).mean()
        vae_loss,_ =self.vae.compute_vae_loss(o_trajectiory[self.agent_id], u_prev_trajectiory[self.agent_id], u_trajectiory[self.agent_id], r_trajectiory.unsqueeze(dim=2), o_next_trajectiory[self.agent_id])

        if self.train_step % 2 == 0:
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            #
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

        else:
            self.vae.optimiser_vae.zero_grad()
            vae_loss.backward()
            self.vae.optimiser_vae.step()

        self._soft_update_target_network()
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


