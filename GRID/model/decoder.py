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
                 reward_embed_dim=5
                 ):
        super(ActionDecoder, self).__init__()
        self.args = args
        self.state_encoder = nn.Linear(state_dim, state_embed_dim)
        self.action_encoder = nn.Linear(action_dim, action_embed_dim)
        self.reward_encoder = nn.Linear(reward_dim, reward_embed_dim)

        curr_input_dim = latent_dim + state_embed_dim + action_embed_dim + reward_embed_dim
        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
            curr_input_dim = layers[i]

        # output layer
        self.fc_out = nn.Linear(curr_input_dim, action_dim)

    def forward(self, latent_state, state, action, reward):
        action = F.one_hot(action, num_classes=5).squeeze(dim=2).float()
        state = F.one_hot(state, num_classes=self.args.high*self.args.width).squeeze(dim=2).float()
        ha = F.relu(self.action_encoder(action))
        hs = F.relu(self.state_encoder(state))
        hr = F.relu(self.reward_encoder(reward))

        h = torch.cat((latent_state, hs, ha, hr), dim=2)

        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))

        a = F.gumbel_softmax(self.fc_out(h), hard=False)
        return a