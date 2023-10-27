import torch
import torch.nn.functional as F
import torch.nn as nn
class Encoder(nn.Module):
    def __init__(self,
                 # network size
                 layers_before_gru=(),
                 hidden_size=64,
                 layers_after_gru=(),
                 latent_dim=32,
                 action_dim=5,
                 action_embed_dim=10,
                 state_dim=49,
                 state_embed_dim=10,
                 reward_dim=1,
                 reward_embed_dim=5
                 ):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.state_dim = state_dim

        # embed action, state, reward
        self.state_encoder = torch.nn.Linear(state_dim, state_embed_dim)
        self.action_encoder = torch.nn.Linear(action_dim, action_embed_dim)
        self.reward_encoder = torch.nn.Linear(reward_dim, reward_embed_dim)

        # fully connected layers before the recurrent cell
        curr_input_dim = action_embed_dim + state_embed_dim + reward_embed_dim
        self.fc_before_gru_model = torch.nn.Sequential()
        for i in range(len(layers_before_gru)):
            self.fc_before_gru_model.add_module(torch.nn.Linear(curr_input_dim, layers_before_gru[i]))
            self.fc_before_gru_model.add_module(F.relu())
            curr_input_dim = layers_before_gru[i]

        # recurrent unit
        self.gru_model = torch.nn.GRU(curr_input_dim, hidden_size, batch_first=True)

        # fully connected layers after the recurrent cell
        curr_input_dim = hidden_size
        self.fc_after_gru_model = torch.nn.Sequential()
        for i in range(len(layers_after_gru)):
            self.fc_after_gru_model.add_module(torch.nn.Linear(curr_input_dim, layers_after_gru[i]))
            self.fc_after_gru_model.add_module(F.relu())
            curr_input_dim = layers_after_gru[i]

        # output layer
        self.fc_mu = torch.nn.Linear(curr_input_dim, latent_dim)
        self.fc_logvar = torch.nn.Linear(curr_input_dim, latent_dim)

    def sample_gaussian(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        output = torch.randn_like(std)
        return output.mul(std).add_(mu)

    def forward(self, states, prev_actions, rewards, sample=True):
        """
        Actions, states, rewards should be given in form [sequence_len * batch_size * dim].
        For one-step predictions, sequence_len=1 and hidden_state!=None.
        For feeding in entire trajectories, sequence_len>1 and hidden_state=None.
        In the latter case, we return embeddings of length sequence_len+1 since they include the prior.
        """
        # shape should be: sequence_len x batch_size x hidden_size
        prev_actions = prev_actions.reshape((-1, *prev_actions.shape[-2:]))
        states = states.reshape((-1, *states.shape[-2:]))
        rewards = rewards.reshape((-1, *rewards.shape[-2:]))

        states = F.one_hot(states, num_classes=self.state_dim).squeeze(dim=2).float()
        prev_actions = F.one_hot(prev_actions, num_classes=5).squeeze(dim=2).float()

        hpa = F.relu(self.action_encoder(prev_actions))
        hs = F.relu(self.state_encoder(states))
        hr = F.relu(self.reward_encoder(rewards))

        h = torch.cat((hs, hpa, hr), dim=2)

        # forward through fully connected layers before GRU
        h = self.fc_before_gru_model(h)
        # GRU cell (output is outputs for each time step, hidden_state is last output)
        output, _ = self.gru_model(h)
        gru_h = output
        # forward through fully connected layers after GRU
        gru_h = self.fc_after_gru_model(gru_h)

        # outputs
        latent_mean = self.fc_mu(gru_h)
        latent_logvar = self.fc_logvar(gru_h)

        if sample:
            latent_sample = self.sample_gaussian(latent_mean, latent_logvar)
        else:
            latent_sample = latent_mean
        return latent_sample, latent_mean, latent_logvar
