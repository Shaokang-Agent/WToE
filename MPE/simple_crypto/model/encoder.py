import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
# import sys
#
# sys.path.append('../')
# import utils

class Encoder(nn.Module):
    def __init__(self,
                 # network size
                 layers_before_gru=(),
                 hidden_size=64,
                 layers_after_gru=(),
                 latent_dim=32,
                 action_dim=5,
                 action_embed_dim=10,
                 state_dim=16,
                 state_embed_dim=10,
                 reward_dim=1,
                 reward_embed_dim=5,
                 distribution='gaussian',
                 ):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_size = hidden_size

        if distribution == 'gaussian':
            self.reparameterise = self.sample_gaussian
        else:
            raise NotImplementedError

        # embed action, state, reward
        self.state_encoder = torch.nn.Linear(state_dim, state_embed_dim)
        self.action_encoder = torch.nn.Linear(action_dim, action_embed_dim)
        self.reward_encoder = torch.nn.Linear(reward_dim, reward_embed_dim)

        # fully connected layers before the recurrent cell
        curr_input_dim = action_embed_dim + 2 * state_embed_dim + reward_embed_dim
        self.fc_before_gru_model = torch.nn.Sequential()
        for i in range(len(layers_before_gru)):
            self.fc_before_gru_model.add_module(torch.nn.Linear(curr_input_dim, layers_before_gru[i]))
            self.fc_before_gru_model.add_module(F.relu())
            curr_input_dim = layers_before_gru[i]

        # recurrent unit
        self.gru_model = torch.nn.GRU(curr_input_dim, hidden_size)

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

    # def reset_hidden(self, hidden_state, reset_task):
    #     if hidden_state.dim() != reset_task.dim():
    #         if reset_task.dim() == 2:
    #             reset_task = reset_task.unsqueeze(0)
    #         elif reset_task.dim() == 1:
    #             reset_task = reset_task.unsqueeze(0).unsqueeze(2)
    #     hidden_state = hidden_state * (1 - reset_task)
    #     return hidden_state

    # def prior(self, batch_size, sample=True):
    #
    #     # TODO: add option to incorporate the initial state
    #
    #     # we start out with a hidden state of zero
    #     hidden_state = tf.zeros((batch_size, self.hidden_size))
    #     h = self.fc_after_gru_model(hidden_state)
    #     # forward through fully connected layers after GRU
    #
    #     # outputs
    #     latent_mean = self.fc_mu(h)
    #     latent_logvar = self.fc_logvar(h)
    #     if sample:
    #         latent_sample = self.reparameterise(latent_mean, latent_logvar)
    #     else:
    #         latent_sample = latent_mean
    #
    #     return latent_sample, latent_mean, latent_logvar, hidden_state


    def forward(self, states, next_states, prev_actions, rewards, sample=True):
        prev_actions = prev_actions.reshape((-1, *prev_actions.shape[-2:]))
        states = states.reshape((-1, *states.shape[-2:]))
        next_states = next_states.reshape((-1, *next_states.shape[-2:]))
        rewards = rewards.reshape((-1, *rewards.shape[-2:]))


        hpa = F.relu(self.action_encoder(prev_actions))
        hs = F.relu(self.state_encoder(states))
        hns = F.relu(self.state_encoder(next_states))
        hr = F.relu(self.reward_encoder(rewards))

        h = torch.cat((hs, hns, hpa, hr), dim=2)

        # forward through fully connected layers before GRU
        h = self.fc_before_gru_model(h)
        # GRU cell (output is outputs for each time step, hidden_state is last output)
        output, _ = self.gru_model(h)

        # gru_h = F.relu(output)  # TODO: should this be here?
        gru_h = output

        # forward through fully connected layers after GRU
        gru_h = self.fc_after_gru_model(gru_h)

        # outputs
        latent_mean = self.fc_mu(gru_h)
        latent_logvar = self.fc_logvar(gru_h)

        if sample:
            latent_sample = self.reparameterise(latent_mean, latent_logvar)
        else:
            latent_sample = latent_mean

        # if return_prior:
        #     latent_sample = tf.concat((prior_sample, latent_sample))
        #     latent_mean = tf.concat((prior_mean, latent_mean))
        #     latent_logvar = tf.concat((prior_logvar, latent_logvar))
        #     output = tf.concat((prior_hidden_state, output))

        if latent_mean.shape[0] == 1:
            latent_sample, latent_mean, latent_logvar = latent_sample[0], latent_mean[0], latent_logvar[0]

        return latent_sample, latent_mean, latent_logvar
