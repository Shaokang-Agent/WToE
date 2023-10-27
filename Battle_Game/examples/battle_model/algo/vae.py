import numpy as np
from . import encoder
from . import decoder
import tensorflow as tf
from random import sample

class VAE(object):
    def __init__(self, sess, name, handle, env):
        # initialise the encoder
        self.sess = sess
        self.encoder = encoder.Encoder(
            sess,
            name,
            handle,
            env,
            hidden_size = 32,
            latent_dim = 32,
            action_embed_dim = 32,
            state_embed_dim = 64,
            reward_embed_dim = 5,
        )
        # initialise the decoders (returns None for unused decoders)
        self.action_decoder = decoder.ActionDecoder(
            sess,
            name,
            handle,
            env,
            latent_dim=32,
            action_embed_dim=32,
            state_embed_dim=64,
            reward_embed_dim=5,
        )

    def train(self, episode_buffer):
        views = []
        features = []
        actions = []
        rewards = []
        Batch = len(list(episode_buffer))
        for i in range(Batch):
            padding_step_num=200
            padding_views = np.zeros([padding_step_num,13,13,7])
            padding_features = np.zeros([padding_step_num,34])
            padding_actions = np.zeros([padding_step_num])
            padding_rewards = np.zeros([padding_step_num])
            views_length = min(padding_step_num, len(list(episode_buffer)[i].views))
            padding_views[:views_length] = np.array(list(episode_buffer)[i].views[:views_length])
            views.append(padding_views)
            features_length = min(padding_step_num, len(list(episode_buffer)[i].features))
            padding_features[:features_length] = np.array(list(episode_buffer)[i].features[:features_length])
            features.append(padding_features)
            actions_length = min(padding_step_num, len(list(episode_buffer)[i].actions))
            padding_actions[:actions_length] = np.array(list(episode_buffer)[i].actions[:actions_length])
            actions.append(padding_actions)
            rewards_length = min(padding_step_num, len(list(episode_buffer)[i].rewards))
            padding_rewards[:rewards_length] = np.array(list(episode_buffer)[i].rewards[:rewards_length])
            rewards.append(padding_rewards)
        views = np.array(views)
        features = np.array(features)
        actions = np.array(actions)
        rewards = np.array(rewards)
        Step = views.shape[1]
        #print(Batch, Step, views.shape, features.shape, actions.shape, rewards.shape)
        h_obs_epi = []
        for i in range(1, Step):
            h_obs = self.sess.run(self.encoder.h_obs, feed_dict={self.encoder.obs_input: views[:,i,:]})
            h_obs_epi.append(h_obs)

        h_obs_epi = np.array(h_obs_epi).transpose(1,0,2)
        self.pre_gru_state = np.zeros([Batch, 32])
        output_sample, fc_mu, fc_logvar, _ = self.encoder.output_sample()
        output_sample, fc_mu, fc_logvar = self.sess.run([output_sample, fc_mu, fc_logvar], feed_dict={
            self.encoder.h_obs_aft_cnn: h_obs_epi,
            self.encoder.feat_input: features[:, 1:, :],
            self.encoder.act_input: np.expand_dims(actions[:, :-1], axis=2),
            self.encoder.reward_input: np.expand_dims(rewards[:, :-1], axis=2),
            self.encoder.pre_gru_state: self.pre_gru_state
        })

        action_pred = self.action_decoder.output_pred_action()
        action_label = tf.placeholder(tf.int32, (None,None,1), name="Act-Label")
        loss_func1 = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=action_label, logits=action_pred))
        loss_func2 = tf.reduce_mean(- 0.5 * (1 + fc_logvar - tf.square(fc_mu) - tf.exp(fc_logvar)))
        loss = 0.1 * loss_func1 + loss_func2
        train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
        self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        h_obs_epi = []
        for i in range(1, Step):
            h_obs = self.sess.run(self.action_decoder.h_obs, feed_dict={self.action_decoder.obs_input: views[:,i,:]})
            h_obs_epi.append(h_obs)

        h_obs_epi = np.array(h_obs_epi).transpose(1, 0, 2)
        self.sess.run(train_op, feed_dict={
            self.action_decoder.h_obs_aft_cnn: h_obs_epi,
            self.action_decoder.feat_input: features[:, 1:, :],
            self.action_decoder.act_input: np.expand_dims(actions[:, :-1], axis=2),
            self.action_decoder.reward_input: np.expand_dims(rewards[:, :-1], axis=2),
            self.action_decoder.z: output_sample,
            action_label: np.expand_dims(actions[:, 1:], axis=2)
        })



