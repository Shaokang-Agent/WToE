import numpy as np
import tensorflow as tf

class ActionDecoder():
    def __init__(self, sess, name, handle, env, latent_dim, action_embed_dim, state_embed_dim, reward_embed_dim):
        super(ActionDecoder, self).__init__()
        self.sess = sess
        self.latent_dim = latent_dim
        self.view_space = env.get_view_space(handle)
        assert len(self.view_space) == 3
        self.feature_space = env.get_feature_space(handle)
        self.num_actions = env.get_action_space(handle)[0]
        self.name = name

        # embed action, state, reward
        with tf.variable_scope(name):
            self.feat_input = tf.placeholder(tf.float32, (None,None,) + self.feature_space, name="de-Feat-Input")
            self.act_input = tf.placeholder(tf.int32, (None,None,1), name="de-Act-Input")
            self.act_one_hot = tf.squeeze(tf.one_hot(self.act_input, depth=self.num_actions, on_value=1.0, off_value=0.0), axis=2)
            self.reward_input = tf.placeholder(tf.float32, (None,None,1), name="de-Reward-Input")

            self.obs_input = tf.placeholder(tf.float32, (None,) + self.view_space, name="de-Obs-CNN-Input")
            self.conv1 = tf.layers.conv2d(self.obs_input, filters=32, kernel_size=3, activation=tf.nn.relu, name="de-Conv1")
            self.conv2 = tf.layers.conv2d(self.conv1, filters=32, kernel_size=3, activation=tf.nn.relu, name="de-Conv2")
            self.flatten_obs = tf.reshape(self.conv2, [-1, np.prod([v.value for v in self.conv2.shape[1:]])])
            self.h_obs = tf.layers.dense(self.flatten_obs, units=256, activation=tf.nn.relu, name="de-Dense-Obs")

            self.h_obs_aft_cnn = tf.placeholder(tf.float32, (None, None, 256), name="de-Obs-Input")
            self.h_emb = tf.layers.dense(self.feat_input, units=32, activation=tf.nn.relu, name="de-Dense-Emb")
            self.concat_layer = tf.concat([self.h_obs_aft_cnn, self.h_emb], axis=2)

            self.state_encoder = tf.layers.dense(self.concat_layer, units=state_embed_dim, activation=tf.nn.relu,
                                                 name="de-state-encoder")
            self.action_encoder = tf.layers.dense(self.act_one_hot, units=action_embed_dim, activation=tf.nn.relu,
                                                  name="de-action-encoder")
            self.reward_encoder = tf.layers.dense(self.reward_input, units=reward_embed_dim, activation=tf.nn.relu,
                                                  name="de-reward-encoder")

            self.z = tf.placeholder(tf.float32, (None,None,self.latent_dim), name="de-z-input")
            self.decoder_input = tf.concat([self.z, self.state_encoder, self.action_encoder, self.reward_encoder],axis=2)
            self.fc_layers = tf.layers.dense(self.decoder_input, units=32, activation=tf.nn.relu, name="de-fc_layers")
            self.fc_out = tf.layers.dense(self.fc_layers, units=self.num_actions, activation=tf.nn.softmax,name="de-fc_out")

    def output_pred_action(self):
        return self.fc_out
