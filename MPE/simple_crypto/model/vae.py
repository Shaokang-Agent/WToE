import gym
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


from model.decoder import ActionDecoder
from model.encoder import Encoder


class VAE:
    def __init__(self, args, agent_id):
        self.args = args
        self.agent_id = agent_id
        # initialise the encoder
        self.encoder = self.initialise_encoder()
        # initialise the decoders (returns None for unused decoders)
        self.action_decoder = self.initialise_decoder()

        if self.args.cuda:
            self.encoder.cuda()
            self.action_decoder.cuda()
        # initalise optimiser for the encoder and decoders
        self.optimiser_vae = torch.optim.Adam([*self.encoder.parameters(), *self.action_decoder.parameters()], lr=self.args.lr_vae)

    def initialise_decoder(self):
        latent_dim = self.args.latent_dim
        action_decoder = ActionDecoder(
            args=self.args,
            latent_dim=latent_dim,
            layers=self.args.decoder_layers,
            action_dim=self.args.action_shape[self.agent_id],
            action_embed_dim=self.args.action_embedding_size,
            state_dim=self.args.obs_shape[self.agent_id],
            state_embed_dim=self.args.state_embedding_size,
        )
        return action_decoder

    def initialise_encoder(self):
        encoder = Encoder(
            layers_before_gru=self.args.layers_before_gru,
            hidden_size=self.args.gru_hidden_size,
            layers_after_gru=self.args.layers_after_gru,
            latent_dim=self.args.latent_dim,
            action_dim=self.args.action_shape[self.agent_id],
            action_embed_dim=self.args.action_embedding_size,
            state_dim=self.args.obs_shape[self.agent_id],
            state_embed_dim=self.args.state_embedding_size,
            reward_dim=1,
            reward_embed_dim=self.args.reward_embedding_size,
        )
        return encoder

    def compute_action_reconstruction_loss(self, embedding, obs, next_obs, prev_actions, reward, actions, return_predictions=True):
        action_reconstruction = self.action_decoder(embedding, obs, prev_actions, next_obs, reward)
        loss_action = (action_reconstruction - actions).pow(2).mean()
        if return_predictions:
            return loss_action, action_reconstruction
        else:
            return loss_action

    def compute_kl_loss(self, latent_mean, latent_logvar):
        kl_divergences = - 0.5 * (1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp()).mean()
        return kl_divergences

    def compute_vae_loss(self, o_trajectiory, u_prev_trajectiory, u_trajectiory, r_trajectiory, o_next_trajectiory):
        output_embedding, mean, logvar = self.encoder(prev_actions=u_prev_trajectiory,
                                                      states=o_trajectiory,
                                                      rewards=r_trajectiory,
                                                      next_states=o_next_trajectiory,
                                                      sample=True)
        loss_kl = self.compute_kl_loss(mean, logvar)
        loss_action, action_reconstruction = self.compute_action_reconstruction_loss(output_embedding, o_trajectiory, o_next_trajectiory, u_prev_trajectiory, r_trajectiory, u_trajectiory)
        elbo_loss = loss_kl + loss_action
        return elbo_loss, action_reconstruction


    # def log(self, elbo_loss, rew_reconstruction_loss, state_reconstruction_loss, task_reconstruction_loss, kl_loss):
    #
    #     curr_iter_idx = self.get_iter_idx()
    #     if curr_iter_idx % self.args.log_interval == 0:
    #
    #         if self.args.decode_reward:
    #             self.logger.add('vae_losses/reward_reconstr_err', rew_reconstruction_loss.mean(), curr_iter_idx)
    #         if self.args.decode_state:
    #             self.logger.add('vae_losses/state_reconstr_err', state_reconstruction_loss.mean(), curr_iter_idx)
    #         if self.args.decode_task:
    #             self.logger.add('vae_losses/task_reconstr_err', task_reconstruction_loss.mean(), curr_iter_idx)
    #
    #         if not self.args.disable_stochasticity_in_latent:
    #             self.logger.add('vae_losses/kl', kl_loss.mean(), curr_iter_idx)
    #         self.logger.add('vae_losses/sum', elbo_loss, curr_iter_idx)
