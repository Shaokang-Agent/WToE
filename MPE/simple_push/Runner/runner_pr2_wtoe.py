from tqdm import tqdm
from Agent.WToE_agent import WToE_Agent
from Agent.pr2_agent import PR2_Agent
from common.WToE_repaly_buffer import WToE_Buffer
from common.replay_buffer import Buffer
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import copy
import argparse

class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.agents = self._init_agents()
        self.buffer = Buffer(args)

        self.save_path = self.args.save_dir + '/' + self.args.scenario_name + '/' + self.args.algorithm
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        #adv setting
        self.adv_args = argparse.Namespace(**vars(self.args))
        self.adv_args.n_players = self.env.n
        self.adv_args.n_agents = 1
        self.adv_args.obs_shape = [self.env.observation_space[self.adv_args.n_players-1-i].shape[0] for i in range(self.adv_args.n_agents)]  # 每一维代表该agent的obs维度
        self.adv_args.action_shape = [self.env.action_space[self.adv_args.n_players-1-i].n for i in range(self.adv_args.n_agents)]
        self.adv_args.high_action = 1
        self.adv_args.low_action = -1
        self.adv_buffer = WToE_Buffer(self.adv_args)

        self.adv_agents = self._init_adv_agents()


    def _init_agents(self):
        print("Good: PR2")
        agents = []
        for i in range(self.args.n_agents):
            agent = PR2_Agent(i, self.args)
            agents.append(agent)
        return agents

    def _init_adv_agents(self):
        print("Adv: WToE")
        agents = []
        for i in range(self.adv_args.n_agents):
            agent = WToE_Agent(i, self.adv_args)
            agents.append(agent)
        return agents

    def run(self):
        returns = []
        s_stack = [[] for _ in range(self.args.n_players - self.args.n_agents)]
        a_stack = [[] for _ in range(self.args.n_players - self.args.n_agents)]
        r_stack = [[] for _ in range(self.args.n_players - self.args.n_agents)]
        for time_step in tqdm(range(self.args.time_steps)):
            # reset the environment
            if time_step % self.episode_limit == 0:
                s = self.env.reset()
            u = []
            actions = []
            with torch.no_grad():

                maddpg_actions = []
                for i in range(self.args.n_agents):
                    action = self.agents[i].select_action(s[i], self.noise, self.epsilon)
                    maddpg_actions.append(action)
                    actions.append(action)

                for i in range(self.args.n_agents, self.args.n_players):
                    agent_id = i - self.args.n_agents
                    if time_step == 0:
                        for index in range(self.args.history_size + 2):
                            s_stack[agent_id].append(s[i])
                    else:
                        s_stack[agent_id].pop(0)
                        s_stack[agent_id].append(s[i])
                    action = self.adv_agents[agent_id].select_action(s[i], self.noise, self.epsilon, time_step,
                                                                     copy.deepcopy(s_stack[agent_id]),
                                                                     copy.deepcopy(a_stack[agent_id]),
                                                                     copy.deepcopy(r_stack[agent_id]))
                    if time_step == 0:
                        for index in range(self.args.history_size + 2):
                            a_stack[agent_id].append(action)
                    else:
                        a_stack[agent_id].pop(0)
                        a_stack[agent_id].append(action)
                    u.append(action)
                    actions.append(action)

            s_next, r, done, info = self.env.step(actions)
            for i in range(self.args.n_agents, self.args.n_players):
                agent_id = i - self.args.n_agents
                if time_step == 0:
                    for index in range(self.args.history_size + 2):
                        r_stack[agent_id].append(r[agent_id])
                else:
                    r_stack[agent_id].pop(0)
                    r_stack[agent_id].append(r[agent_id])
            self.adv_buffer.store_episode(s[self.args.n_agents:], u, r[self.args.n_agents:], s_next[self.args.n_agents:])

            self.buffer.store_episode(s[:self.args.n_agents], maddpg_actions, r[:self.args.n_agents], s_next[:self.args.n_agents])
            s = s_next
            if self.buffer.current_size >= self.args.batch_size:
                transitions = self.buffer.sample(self.args.batch_size)
                for agent in self.agents:
                    other_agents = self.agents.copy()
                    other_agents.remove(agent)
                    agent.learn(transitions, other_agents)

                adv_transitions = self.adv_buffer.sample(self.args.batch_size)
                for agent in self.adv_agents:
                    other_agents = self.adv_agents.copy()
                    other_agents.remove(agent)
                    agent.learn(adv_transitions, other_agents)

            if time_step > 0 and time_step % self.args.evaluate_rate == 0:
                returns.append(self.evaluate())
                plt.figure()
                plt.plot(range(len(returns)), returns)
                plt.xlabel('episode * ' + str(self.args.evaluate_rate / self.episode_limit))
                plt.ylabel('average returns')
                plt.savefig(self.save_path + '/plt.png', format='png')
            self.noise = max(0.05, self.noise - 0.0000005)
            self.epsilon = max(0.05, self.noise - 0.0000005)
            np.save(self.save_path + '/returns.pkl', returns)

    def evaluate(self):
        returns_good = []
        returns_adv = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.reset()
            rewards_good = 0
            rewards_adv = 0
            for time_step in range(self.args.evaluate_episode_len):
                #self.env.render()
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], 0, 0)
                        actions.append(action)
                    for i in range(self.args.n_agents, self.args.n_players):
                        action = self.adv_agents[i-self.args.n_agents].select_action(s[i], 0, 0)
                        actions.append(action)
                s_next, r, done, info = self.env.step(actions)
                rewards_good += r[0]
                rewards_adv += r[-1]
                s = s_next
            returns_good.append(rewards_good)
            returns_adv.append(rewards_adv)
            print('Good Returns is {}, Adversary Returns is {}'.format(rewards_good, rewards_adv))
        return sum(returns_good) / self.args.evaluate_episodes - sum(returns_adv) / self.args.evaluate_episodes
