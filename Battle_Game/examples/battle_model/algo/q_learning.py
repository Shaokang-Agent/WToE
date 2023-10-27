import os
import tensorflow as tf
import numpy as np
from . import vae
from . import base
from . import tools


class DQN(base.ValueNet):
    def __init__(self, sess, name, handle, env, sub_len, memory_size=2**10,wtoe=False,batch_size=64, update_every=5):
        super().__init__(sess, env, handle, name, update_every=update_every,wtoe=wtoe)
        self.wtoe = wtoe
        self.action_num = env.get_action_space(handle)[0]
        self.train_times = 0
        self.max_train_episodes = 25
        self.max_train_steps = 200
        if self.wtoe:
            self.VAE = vae.VAE(sess, name, handle, env)
        self.replay_buffer = tools.MemoryGroup(self.view_space, self.feature_space, self.num_actions, memory_size, batch_size, sub_len)
        self.episode_buffer = tools.EpisodesBuffer()
        self.sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])

    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)
        if self.train_times <= self.max_train_episodes:
            self.episode_buffer.push(**kwargs)

    def act(self, **kwargs):
        feed_dict = {
            self.obs_input: kwargs['state'][0],
            self.feat_input: kwargs['state'][1]
        }

        self.temperature = kwargs['eps']

        if self.use_mf:
            assert kwargs.get('prob', None) is not None
            assert len(kwargs['prob']) == len(kwargs['state'][0])
            feed_dict[self.act_prob_input] = kwargs['prob']

        actions = self.sess.run(self.predict, feed_dict=feed_dict)
        actions = np.argmax(actions, axis=1).astype(np.int32)

        explore_num = kwargs['explore_num']
        if self.wtoe and kwargs['pre_a'] != [] and kwargs['n_round'] <= self.max_train_episodes and explore_num > 0 and kwargs['step_ct'] <= self.max_train_steps:
            pre_length = kwargs['pre_gru_state'].shape[0]
            if False in kwargs['pre_alives'][:pre_length]:
                death_index = np.where(kwargs['pre_alives'][:pre_length] == False)
                kwargs['pre_a'] = np.delete(kwargs['pre_a'][:pre_length], death_index)
                kwargs['pre_r'] = np.delete(kwargs['pre_r'][:pre_length], death_index)
                kwargs['pre_gru_state'] = np.delete(kwargs['pre_gru_state'], death_index, axis=0)

            h_obs = self.sess.run(self.VAE.encoder.h_obs, feed_dict={self.VAE.encoder.obs_input: kwargs['state'][0][:explore_num]})
            encode_feed_dict = {
                self.VAE.encoder.h_obs_aft_cnn: np.expand_dims(h_obs, axis=1),
                self.VAE.encoder.feat_input: np.expand_dims(np.array(kwargs['state'][1][:explore_num]), axis=1),
                self.VAE.encoder.act_input: np.expand_dims(np.expand_dims(np.array(kwargs['pre_a'][:explore_num]), axis=1), axis=1),
                self.VAE.encoder.reward_input: np.expand_dims(np.expand_dims(np.array(kwargs['pre_r'][:explore_num]), axis=1), axis=1),
                self.VAE.encoder.pre_gru_state: kwargs['pre_gru_state'][:explore_num],
            }
            output_sample, _, _, gru_state = self.VAE.encoder.output_sample()
            output_sample,gru_state = self.sess.run([output_sample,gru_state], feed_dict=encode_feed_dict)
            # print(output_sample.shape,gru_state.shape)
            h_obs = self.sess.run(self.VAE.action_decoder.h_obs, feed_dict={self.VAE.action_decoder.obs_input: kwargs['state'][0][:explore_num]})
            decode_feed_dict = {
                self.VAE.action_decoder.h_obs_aft_cnn: np.expand_dims(h_obs, axis=1),
                self.VAE.action_decoder.feat_input: np.expand_dims(np.array(kwargs['state'][1][:explore_num]), axis=1),
                self.VAE.action_decoder.act_input: np.expand_dims(np.expand_dims(np.array(kwargs['pre_a'][:explore_num]), axis=1), axis=1),
                self.VAE.action_decoder.reward_input: np.expand_dims(np.expand_dims(np.array(kwargs['pre_r'][:explore_num]), axis=1), axis=1),
                self.VAE.action_decoder.z: output_sample,
            }
            short_term_action = self.sess.run(self.VAE.action_decoder.output_pred_action(), feed_dict=decode_feed_dict)
            for i in range(short_term_action.shape[0]):
                short_term_action_sample = np.random.choice(a=range(self.action_num), size=1, replace=True, p=short_term_action[i][0])
                if short_term_action_sample != actions[i]:
                    if np.random.rand() < kwargs['eps']/3:
                        actions[i] = np.random.randint(low=0, high=self.action_num)
            return actions,gru_state
        return actions, kwargs['pre_gru_state']
    
    def train(self):
        if self.wtoe and self.train_times <= self.max_train_episodes:
            list_epi_buffer = self.episode_buffer.episodes()
            self.VAE.train(list_epi_buffer)
            self.episode_buffer.reset()
        self.replay_buffer.tight()
        batch_num = self.replay_buffer.get_batch_num()

        for i in range(batch_num):
            obs, feats, obs_next, feat_next, dones, rewards, actions, masks = self.replay_buffer.sample()
            target_q = self.calc_target_q(obs=obs_next, feature=feat_next, rewards=rewards, dones=dones)
            loss, q = super().train(state=[obs, feats], target_q=target_q, acts=actions, masks=masks)

            self.update()

            if i % 50 == 0:
                print('[*] LOSS:', loss, '/ Q:', q)
        self.train_times += 1

    def save(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)

        file_path = os.path.join(dir_path, "dqn_{}".format(step))
        saver.save(self.sess, file_path)

        print("[*] Model saved at: {}".format(file_path))

    def load(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)

        file_path = os.path.join(dir_path, "dqn_{}".format(step))

        saver.restore(self.sess, file_path)
        print("[*] Loaded model from {}".format(file_path))


