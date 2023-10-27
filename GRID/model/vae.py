import torch
from model.decoder import ActionDecoder
from model.encoder import Encoder


class VAE:
    def __init__(self, args):
        self.args = args
        # initialise the encoder
        self.encoder = self.initialise_encoder()
        # initialise the decoders (returns None for unused decoders)
        self.action_decoder = self.initialise_decoder()
        # initalise optimiser for the encoder and decoders
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.optimiser_vae = torch.optim.Adam([*self.encoder.parameters(), *self.action_decoder.parameters()], lr=self.args.lr_vae)

    def initialise_decoder(self):
        latent_dim = self.args.latent_dim
        action_decoder = ActionDecoder(
            args=self.args,
            latent_dim=latent_dim,
            layers=self.args.decoder_layers,
            action_dim=5,
            action_embed_dim=self.args.action_embedding_size,
            state_dim=self.args.high*self.args.width,
            state_embed_dim=self.args.state_embedding_size,
        )
        return action_decoder

    def initialise_encoder(self):
        encoder = Encoder(
            layers_before_gru=self.args.layers_before_gru,
            hidden_size=self.args.gru_hidden_size,
            layers_after_gru=self.args.layers_after_gru,
            latent_dim=self.args.latent_dim,
            action_dim=5,
            action_embed_dim=self.args.action_embedding_size,
            state_dim=self.args.high*self.args.width,
            state_embed_dim=self.args.state_embedding_size,
            reward_dim=1,
            reward_embed_dim=self.args.reward_embedding_size,
        )
        return encoder

    def compute_action_reconstruction_loss(self, embedding, obs, prev_actions, reward, actions, return_predictions=True):
        action_reconstruction = self.action_decoder(embedding, obs, prev_actions, reward)
        loss_action = self.loss_func(action_reconstruction.reshape(-1,5), actions.squeeze(dim=2).reshape(-1))
        if return_predictions:
            return loss_action, action_reconstruction
        else:
            return loss_action

    def compute_kl_loss(self, latent_mean, latent_logvar):
        kl_divergences = - 0.5 * (1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp()).mean()
        return kl_divergences

    def compute_vae_loss(self, o_trajectiory, u_prev_trajectiory, u_trajectiory, r_trajectiory):
        output_embedding, mean, logvar = self.encoder(prev_actions=u_prev_trajectiory,
                                                      states=o_trajectiory,
                                                      rewards=r_trajectiory,
                                                      sample=True)
        loss_kl = self.compute_kl_loss(mean, logvar)
        loss_action, action_reconstruction = self.compute_action_reconstruction_loss(output_embedding, o_trajectiory, u_prev_trajectiory, r_trajectiory, u_trajectiory)
        elbo_loss = loss_kl + 0.1*loss_action
        return elbo_loss, action_reconstruction, output_embedding

