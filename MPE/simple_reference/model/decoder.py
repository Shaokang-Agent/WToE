import torch
import torch.nn as nn
from torch.nn import functional as F

class ActionDecoder(nn.Module):
    def __init__(self,
                 args,
                 latent_dim=32,
                 layers=(),
                 action_dim=5,
                 action_embed_dim=10,
                 state_dim=16,
                 state_embed_dim=10,
                 reward_dim=1,
                 reward_embed_dim=5,
                 pred_type='deterministic'
                 ):
        super(ActionDecoder, self).__init__()
        self.args = args
        self.state_encoder = nn.Linear(state_dim, state_embed_dim)
        self.action_encoder = nn.Linear(action_dim, action_embed_dim)
        self.reward_encoder = nn.Linear(reward_dim, reward_embed_dim)

        curr_input_dim = latent_dim + 2*state_embed_dim + action_embed_dim + reward_embed_dim
        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
            curr_input_dim = layers[i]

        # output layer
        if pred_type == 'gaussian':
            self.fc_out = nn.Linear(curr_input_dim, 2 * action_dim)
        else:
            self.fc_out = nn.Linear(curr_input_dim, action_dim)

    def forward(self, latent_state, state, action, next_state, reward):
        ha = F.relu(self.action_encoder(action))
        hs = F.relu(self.state_encoder(state))
        hns = F.relu(self.state_encoder(next_state))
        hr = F.relu(self.reward_encoder(reward))
        h = torch.cat((latent_state, hs, hns, ha, hr), dim=-1)

        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))

        a = self.args.high_action * torch.tanh(self.fc_out(h))
        return a