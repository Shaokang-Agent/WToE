from tqdm import tqdm
from Agent.WToE_agent import WToE_Agent
from Agent.maddpg_agent import Maddpg_Agent
from common.WToE_repaly_buffer import WToE_Buffer
from common.replay_buffer import Buffer
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import copy
import time

class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.agents = self._init_agents()
        self.buffer = WToE_Buffer(args)

        self.save_path = self.args.save_dir + '/' + self.args.scenario_name + '/' + self.args.algorithm
        self.time = np.zeros([int(self.args.time_steps / self.episode_limit)])
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = WToE_Agent(i, self.args)
            agents.append(agent)
        return agents

    def run(self):
        returns = []
        s_stack = [[] for _ in range(self.args.n_agents)]
        a_stack = [[] for _ in range(self.args.n_agents)]
        r_stack = [[] for _ in range(self.args.n_agents)]

        for time_step in tqdm(range(self.args.time_steps)):
            # reset the environment
            if time_step % self.episode_limit == 0:
                s = self.env.reset()
            u = []
            actions = []
            with torch.no_grad():
                for agent_id, agent in enumerate(self.agents):
                    if time_step == 0:
                        for index in range(self.args.history_size+2):
                            s_stack[agent_id].append(s[agent_id])
                    else:
                        s_stack[agent_id].pop(0)
                        s_stack[agent_id].append(s[agent_id])
                    action = agent.select_action(s[agent_id], self.noise, self.epsilon, time_step, copy.deepcopy(s_stack[agent_id]), copy.deepcopy(a_stack[agent_id]), copy.deepcopy(r_stack[agent_id]))
                    if time_step == 0:
                        for index in range(self.args.history_size+2):
                            a_stack[agent_id].append(action)
                    else:
                        a_stack[agent_id].pop(0)
                        a_stack[agent_id].append(action)
                    u.append(action)
                    actions.append(action)
            for i in range(self.args.n_agents, self.args.n_players):
                actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
            s_next, r, done, info = self.env.step(actions)
            for agent_id, agent in enumerate(self.agents):
                if time_step == 0:
                    for index in range(self.args.history_size + 2):
                        r_stack[agent_id].append(r[agent_id])
                else:
                    r_stack[agent_id].pop(0)
                    r_stack[agent_id].append(r[agent_id])
            self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents])
            s = s_next
            if self.buffer.current_size >= self.args.batch_size:
                transitions = self.buffer.sample(self.args.batch_size)
                T1 = time.clock()
                for agent in self.agents:
                    other_agents = self.agents.copy()
                    other_agents.remove(agent)
                    agent.learn(transitions, other_agents)
                T2 = time.clock()
                self.time[int(time_step/self.episode_limit)] += T2-T1

            if time_step > 0 and time_step % self.args.evaluate_rate == 0:
                returns.append(self.evaluate())
                # plt.figure()
                # plt.plot(range(len(returns)), returns)
                # plt.xlabel('episode * ' + str(self.args.evaluate_rate / self.episode_limit))
                # plt.ylabel('average returns')
                # plt.savefig(self.save_path + '/plt.png', format='png')
            self.noise = max(0.05, self.noise - 0.0000005)
            self.epsilon = max(0.05, self.noise - 0.0000005)
            #np.save(self.save_path + '/returns.npy', returns)
            np.save(self.save_path + '/time.npy', self.time)

    def evaluate(self):
        returns = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.reset()
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                #self.env.render()
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], 0, 0)
                        actions.append(action)
                for i in range(self.args.n_agents, self.args.n_players):
                    actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                s_next, r, done, info = self.env.step(actions)
                rewards += r[0]
                s = s_next
            returns.append(rewards)
            print('Returns is', rewards)
        return sum(returns) / self.args.evaluate_episodes
