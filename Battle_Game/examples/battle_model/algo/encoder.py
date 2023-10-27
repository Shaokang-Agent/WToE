import numpy as np
import tensorflow as tf

class Encoder():
    def __init__(self, sess, name, handle, env, hidden_size, latent_dim, action_embed_dim, state_embed_dim, reward_embed_dim):
        super(Encoder, self).__init__()
        self.sess = sess
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.view_space = env.get_view_space(handle)
        assert len(self.view_space) == 3
        self.feature_space = env.get_feature_space(handle)
        self.num_actions = env.get_action_space(handle)[0]
        self.name = name
        self.action_embed_dim = action_embed_dim
        self.state_embed_dim = state_embed_dim
        self.reward_embed_dim = reward_embed_dim

        with tf.variable_scope(name):
            self.feat_input = tf.placeholder(tf.float32, (None,None,) + self.feature_space, name="en-Feat-Input")
            self.act_input = tf.placeholder(tf.int32, (None,None,1), name="en-Act-Input")
            self.act_one_hot = tf.squeeze(tf.one_hot(self.act_input, depth=self.num_actions, on_value=1.0, off_value=0.0), axis=2)
            self.reward_input = tf.placeholder(tf.float32, (None,None,1), name="en-Reward-Input")
            self.pre_gru_state = tf.placeholder(tf.float32, (None,hidden_size), name="en-hidden-Input")

            self.obs_input = tf.placeholder(tf.float32, (None,) + self.view_space, name="en-Obs-CNN-Input")
            self.conv1 = tf.layers.conv2d(self.obs_input, filters=32, kernel_size=3, activation=tf.nn.relu, name="en-Conv1")
            self.conv2 = tf.layers.conv2d(self.conv1, filters=32, kernel_size=3, activation=tf.nn.relu,name="en-Conv2")
            self.flatten_obs = tf.reshape(self.conv2, [-1, np.prod([v.value for v in self.conv2.shape[1:]])])
            self.h_obs = tf.layers.dense(self.flatten_obs, units=256, activation=tf.nn.relu,name="en-Dense-Obs")

            self.h_obs_aft_cnn = tf.placeholder(tf.float32, (None,None,256), name="en-Obs-Input")
            self.h_emb = tf.layers.dense(self.feat_input, units=32, activation=tf.nn.relu,name="en-Dense-Emb")
            self.concat_layer = tf.concat([self.h_obs_aft_cnn, self.h_emb], axis=2)
            self.state_encoder = tf.layers.dense(self.concat_layer, units=self.state_embed_dim, activation=tf.nn.relu,name="en-state-encoder")
            self.action_encoder = tf.layers.dense(self.act_one_hot, units=self.action_embed_dim, activation=tf.nn.relu,name="en-action-encoder")
            self.reward_encoder = tf.layers.dense(self.reward_input, units=self.reward_embed_dim, activation=tf.nn.relu,name="en-reward-encoder")

            # recurrent unit
            self.sar_concat = tf.concat([self.state_encoder, self.action_encoder, self.reward_encoder], axis=2)
            cell = tf.nn.rnn_cell.GRUCell(num_units=self.hidden_size, activation=tf.nn.relu, name="en-gru-encoder")
            self.gru_output, self.gru_state = tf.nn.dynamic_rnn(cell, inputs=self.sar_concat,initial_state=self.pre_gru_state, time_major=False)
            # output layer
            self.fc_mu = tf.layers.dense(self.gru_output, units=self.latent_dim, activation=tf.nn.relu, name="en-mu")
            self.fc_logvar = tf.layers.dense(self.gru_output, units=self.latent_dim, activation=tf.nn.relu, name="en-logvar")

    def output_sample(self):
        std = tf.exp(0.5 * self.fc_logvar)
        output_sample = self.fc_mu + std*tf.random_normal(shape=tf.shape(std))
        return output_sample, self.fc_mu, self.fc_logvar, self.gru_state
