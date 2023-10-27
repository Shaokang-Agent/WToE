import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

# define the actor network
class Opponent_Conditional_Policy(nn.Module):
    def __init__(self, args, agent_id):
        super(Opponent_Conditional_Policy, self).__init__()
        self.args = args
        self.max_action = args.high_action
        self.fco = nn.Linear(args.obs_shape[agent_id], 64)
        self.fca = nn.Linear(args.action_shape[agent_id], 64)
        self.fcn = nn.Linear(args.obs_shape[agent_id], 64)

        self.fc = nn.Linear(64, 64)
        self.out = nn.Linear(64, args.action_shape[agent_id]* (args.n_agents - 1))

    def forward(self, o, a):
        normal = Normal(torch.zeros([o.shape[1]]), torch.ones([o.shape[1]]))
        latent = normal.sample().unsqueeze(dim=0)
        if self.args.cuda:
            latent = latent.cuda()
        hsa = F.relu(self.fco(o)) + F.relu(self.fca(a)) + F.relu(self.fcn(latent))
        x = F.relu(self.fc(hsa))
        actions = self.max_action * torch.tanh(self.out(x))
        return actions

class Condition_Policy(nn.Module):
    def __init__(self, args, agent_id):
        super(Condition_Policy, self).__init__()
        self.args = args
        self.max_action = args.high_action
        self.fco = nn.Linear(args.obs_shape[agent_id], 64)
        self.fca = nn.Linear(args.action_shape[agent_id] * (args.n_agents - 1), 64)

        self.fc = nn.Linear(64, 64)
        self.out = nn.Linear(64, args.action_shape[agent_id])

    def forward(self, o, a_other):
        hsa = F.relu(self.fco(o)) + F.relu(self.fca(a_other))
        x = F.relu(self.fc(hsa))
        actions = self.max_action * torch.tanh(self.out(x))
        return actions


class Level_K_Actor(nn.Module):
    def __init__(self, args, agent_id):
        super(Level_K_Actor, self).__init__()
        self.args = args
        self.max_action = args.high_action
        self.agent_id = agent_id
        self.k = args.level_k
        self.fc1 = nn.Linear(args.obs_shape[agent_id], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.action_out = nn.Linear(64, args.action_shape[agent_id])

    def forward(self, o, infer_self_policy, infer_other_policy):
        all_actions = []
        action = None

        for i in range(self.args.level_k + 1):
            if i == 0:
                if self.args.level_k % 2 == 0:
                    action = (2 * torch.rand(self.args.action_shape[self.agent_id]) - 1) * self.max_action
                else:
                    action = (2 * torch.rand(self.args.action_shape[self.agent_id]*(self.args.n_agents-1)) - 1) * self.max_action
                if self.args.cuda:
                    action = action.cuda()
            else:
                if self.args.level_k % 2 == 0:
                    if i % 2 == 1:
                        action = infer_other_policy(o, action).detach()
                    else:
                        action = infer_self_policy(o, action)
                else:
                    if i % 2 == 1:
                        action = infer_self_policy(o, action)
                    else:
                        action = infer_other_policy(o, action).detach()
            all_actions.append(action)

        return action, all_actions

class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.args = args
        self.max_action = args.high_action
        self.fc1 = nn.Linear(sum(args.obs_shape) + sum(args.action_shape), 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.q_out = nn.Linear(64, 1)

    def forward(self, state, action):
        for i in range(len(action)):
            action[i] /= self.max_action
        state = state.reshape(self.args.batch_size, -1)
        action = action.permute(1, 0, 2).reshape(self.args.batch_size, -1)
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value
