import os
import numpy as np
import tensorflow as tf

from . import tools
from . import vae

class ActorCritic:
    def __init__(self, sess, name, handle, env, value_coef=0.1, ent_coef=0.08, gamma=0.95, batch_size=64, learning_rate=1e-4, wtoe=False):
        self.sess = sess
        self.env = env
        self.action_num = env.get_action_space(handle)[0]
        self.name = name
        self.view_space = env.get_view_space(handle)
        self.feature_space = env.get_feature_space(handle)
        self.num_actions = env.get_action_space(handle)[0]
        self.gamma = gamma
        self.wtoe=wtoe
        self.train_times = 0
        self.max_train_episodes = 25
        self.max_train_steps = 200

        if self.wtoe:
            self.VAE = vae.VAE(sess, name, handle, env)
        self.episode_buffer = tools.EpisodesBuffer()

        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.value_coef = value_coef  # coefficient of value in the total loss
        self.ent_coef = ent_coef  # coefficient of entropy in the total loss

        # init training buffers
        self.view_buf = np.empty((1,) + self.view_space)
        self.feature_buf = np.empty((1,) + self.feature_space)
        self.action_buf = np.empty(1, dtype=np.int32)
        self.reward_buf = np.empty(1, dtype=np.float32)
        self.replay_buffer = tools.EpisodesBuffer()

        with tf.variable_scope(name):
            self.name_scope = tf.get_variable_scope().name
            self._create_network(self.view_space, self.feature_space)
    
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name_scope)
    
    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)
        if self.train_times <= self.max_train_episodes:
            self.episode_buffer.push(**kwargs)

    def act(self, **kwargs):
        action = self.sess.run(self.calc_action, {
            self.input_view: kwargs['state'][0],
            self.input_feature: kwargs['state'][1]
        })

        explore_num = kwargs['explore_num']
        if self.wtoe and kwargs['pre_a'] != [] and kwargs['n_round'] <= self.max_train_episodes and explore_num > 0 and kwargs['step_ct'] <= self.max_train_steps:
            pre_length = kwargs['pre_gru_state'].shape[0]
            if False in kwargs['pre_alives'][:pre_length]:
                death_index = np.where(kwargs['pre_alives'][:pre_length] == False)
                kwargs['pre_a'] = np.delete(kwargs['pre_a'][:pre_length], death_index)
                kwargs['pre_r'] = np.delete(kwargs['pre_r'][:pre_length], death_index)
                kwargs['pre_gru_state'] = np.delete(kwargs['pre_gru_state'], death_index, axis=0)

            h_obs = self.sess.run(self.VAE.encoder.h_obs,
                                  feed_dict={self.VAE.encoder.obs_input: kwargs['state'][0][:explore_num]})
            encode_feed_dict = {
                self.VAE.encoder.h_obs_aft_cnn: np.expand_dims(h_obs, axis=1),
                self.VAE.encoder.feat_input: np.expand_dims(np.array(kwargs['state'][1][:explore_num]), axis=1),
                self.VAE.encoder.act_input: np.expand_dims(np.expand_dims(np.array(kwargs['pre_a'][:explore_num]), axis=1),
                                                           axis=1),
                self.VAE.encoder.reward_input: np.expand_dims(
                    np.expand_dims(np.array(kwargs['pre_r'][:explore_num]), axis=1), axis=1),
                self.VAE.encoder.pre_gru_state: kwargs['pre_gru_state'][:explore_num],
            }
            output_sample, _, _, gru_state = self.VAE.encoder.output_sample()
            output_sample, gru_state = self.sess.run([output_sample, gru_state], feed_dict=encode_feed_dict)
            # print(output_sample.shape,gru_state.shape)
            h_obs = self.sess.run(self.VAE.action_decoder.h_obs,
                                  feed_dict={self.VAE.action_decoder.obs_input: kwargs['state'][0][:explore_num]})
            decode_feed_dict = {
                self.VAE.action_decoder.h_obs_aft_cnn: np.expand_dims(h_obs, axis=1),
                self.VAE.action_decoder.feat_input: np.expand_dims(np.array(kwargs['state'][1][:explore_num]), axis=1),
                self.VAE.action_decoder.act_input: np.expand_dims(
                    np.expand_dims(np.array(kwargs['pre_a'][:explore_num]), axis=1), axis=1),
                self.VAE.action_decoder.reward_input: np.expand_dims(
                    np.expand_dims(np.array(kwargs['pre_r'][:explore_num]), axis=1), axis=1),
                self.VAE.action_decoder.z: output_sample,
            }
            short_term_action = self.sess.run(self.VAE.action_decoder.output_pred_action(), feed_dict=decode_feed_dict)
            for i in range(short_term_action.shape[0]):
                short_term_action_sample = np.random.choice(a=range(self.action_num), size=1, replace=True,
                                                            p=short_term_action[i][0])
                if short_term_action_sample != action[i]:
                    if np.random.rand() < kwargs['eps']/3:
                        action[i] = np.random.randint(low=0, high=self.action_num)
            return action.astype(np.int32).reshape((-1,)), gru_state
        return action.astype(np.int32).reshape((-1,)), kwargs['pre_gru_state']

    def _create_network(self, view_space, feature_space):
        input_view = tf.placeholder(tf.float32, (None,) + view_space)
        input_feature = tf.placeholder(tf.float32, (None,) + feature_space)
        action = tf.placeholder(tf.int32, [None])

        reward = tf.placeholder(tf.float32, [None])

        hidden_size = [256]

        # fully connected
        flatten_view = tf.reshape(input_view, [-1, np.prod([v.value for v in input_view.shape[1:]])])
        h_view = tf.layers.dense(flatten_view, units=hidden_size[0], activation=tf.nn.relu)

        h_emb = tf.layers.dense(input_feature,  units=hidden_size[0], activation=tf.nn.relu)

        dense = tf.concat([h_view, h_emb], axis=1)
        dense = tf.layers.dense(dense, units=hidden_size[0] * 2, activation=tf.nn.relu)

        policy = tf.layers.dense(dense / 0.1, units=self.num_actions, activation=tf.nn.softmax)
        policy = tf.clip_by_value(policy, 1e-10, 1-1e-10)

        self.calc_action = tf.multinomial(tf.log(policy), 1)

        value = tf.layers.dense(dense, units=1)
        value = tf.reshape(value, (-1,))

        action_mask = tf.one_hot(action, self.num_actions)
        advantage = tf.stop_gradient(reward - value)

        log_policy = tf.log(policy + 1e-6)
        log_prob = tf.reduce_sum(log_policy * action_mask, axis=1)

        pg_loss = -tf.reduce_mean(advantage * log_prob)
        vf_loss = self.value_coef * tf.reduce_mean(tf.square(reward - value))
        neg_entropy = self.ent_coef * tf.reduce_mean(tf.reduce_sum(policy * log_policy, axis=1))
        total_loss = pg_loss + vf_loss + neg_entropy

        # train op (clip gradient)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(total_loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))

        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(total_loss)

        self.input_view = input_view
        self.input_feature = input_feature
        self.action = action
        self.reward = reward

        self.policy, self.value = policy, value
        self.train_op = train_op
        self.pg_loss, self.vf_loss, self.reg_loss = pg_loss, vf_loss, neg_entropy
        self.total_loss = total_loss

    def train(self):
        if self.wtoe and self.train_times <= self.max_train_episodes:
            list_epi_buffer = self.episode_buffer.episodes()
            self.VAE.train(list_epi_buffer)
            self.episode_buffer.reset()
        # calc buffer size
        n = 0
        # batch_data = sample_buffer.episodes()
        batch_data = self.replay_buffer.episodes()
        self.replay_buffer = tools.EpisodesBuffer()

        for episode in batch_data:
            n += len(episode.rewards)

        self.view_buf.resize((n,) + self.view_space)
        self.feature_buf.resize((n,) + self.feature_space)
        self.action_buf.resize(n)
        self.reward_buf.resize(n)
        view, feature = self.view_buf, self.feature_buf
        action, reward = self.action_buf, self.reward_buf

        ct = 0
        gamma = self.gamma
        # collect episodes from multiple separate buffers to a continuous buffer
        for episode in batch_data:
            v, f, a, r = episode.views, episode.features, episode.actions, episode.rewards
            m = len(episode.rewards)

            r = np.array(r)

            keep = self.sess.run(self.value, feed_dict={
                self.input_view: [v[-1]],
                self.input_feature: [f[-1]],
            })[0]

            for i in reversed(range(m)):
                keep = keep * gamma + r[i]
                r[i] = keep

            view[ct:ct + m] = v
            feature[ct:ct + m] = f
            action[ct:ct + m] = a
            reward[ct:ct + m] = r
            ct += m

        assert n == ct

        # train
        _, pg_loss, vf_loss, ent_loss, state_value = self.sess.run(
            [self.train_op, self.pg_loss, self.vf_loss, self.reg_loss, self.value], feed_dict={
                self.input_view: view,
                self.input_feature: feature,
                self.action: action,
                self.reward: reward,
            })

        self.train_times += 1
        print('[*] PG_LOSS:', np.round(pg_loss, 6), '/ VF_LOSS:', np.round(vf_loss, 6), '/ ENT_LOSS:', np.round(ent_loss), '/ Value:', np.mean(state_value))

    def save(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)

        file_path = os.path.join(dir_path, "ac_{}".format(step))
        saver.save(self.sess, file_path)

        print("[*] Model saved at: {}".format(file_path))

    def load(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)

        file_path = os.path.join(dir_path, "ac_{}".format(step))

        saver.restore(self.sess, file_path)
        print("[*] Loaded model from {}".format(file_path))