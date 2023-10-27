import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import copy

class EnvGoObstacle(object):
    def __init__(self, single=True, agent_index=-1):
        self.high = 11
        self.width = 5
        self.num_agent = 12 if agent_index == -1 else 1
        self.state = []
        self.goal = []
        self.obstacles = np.zeros((self.high, self.width))
        self.single = single
        self.agent_index = agent_index

    def reset(self):
        self.state = []
        self.goal = []

        self.obstacles[0, 2] = 1
        self.obstacles[2, 2] = 1
        self.obstacles[4, 2] = 1
        self.obstacles[6, 2] = 1
        self.obstacles[8, 2] = 1
        self.obstacles[10, 2] = 1

        self.obstacles[3, 2] = 1
        self.obstacles[3, 3] = 1
        self.obstacles[3, 4] = 1
        self.obstacles[7, 2] = 1
        self.obstacles[7, 3] = 1
        self.obstacles[7, 4] = 1

        if self.single:
            goal = [[1,4], [1,4], [5,4], [5,4], [9,4], [9,4], [5,0], [5,0], [5,0], [5,0], [5,0], [5,0]]
            self.goal.append(goal[self.agent_index])
            state = [random.randint(0, self.high-1), random.randint(0, self.width-1)]
            while state in self.goal or self.obstacles[state[0]][state[1]] == 1:
                state = [random.randint(0, self.high-1), random.randint(0, self.width-1)]
            self.state.append(state)

        else:
            self.goal = [[1, 4], [1, 4], [5, 4], [5, 4], [9, 4], [9, 4], [5, 0], [5, 0], [5, 0], [5, 0], [5, 0], [5, 0]]
            self.state = [[0, 0], [2, 0], [4, 0], [6, 0], [8, 0], [10, 0], [0, 4], [2, 4], [4, 4], [6, 4], [8, 4], [10, 4]]
        return self.state

    def get_env_info(self):
        return 0

    def get_reward(self, state, action_list):
        reward = np.zeros(self.num_agent) - 0.1
        next_state = copy.deepcopy(state)
        for i in range(self.num_agent):
            if action_list[i] == 0:  # move right
                next_state[i][1] = state[i][1] + 1
            elif action_list[i] == 1:  # move left
                next_state[i][1] = state[i][1] - 1
            elif action_list[i] == 2:  # move up
                next_state[i][0] = state[i][0] - 1
            elif action_list[i] == 3:  # move down
                next_state[i][0] = state[i][0] + 1
            elif action_list[i] == 4:  # stay
                pass
        for i in range(self.num_agent):
            other_next_state = next_state[:i] + next_state[i+1:]
            other_state = state[:i] + state[i+1:]
            if next_state[i] == self.goal[i]:
                reward[i] = 1
            elif next_state[i][0] < 0 or next_state[i][0] > self.high-1 or next_state[i][1] < 0 or next_state[i][1] > self.width-1 or self.obstacles[next_state[i][0]][next_state[i][1]] == 1:
                next_state[i] = state[i]
                reward[i] = -0.3
            elif next_state[i] in other_next_state:
                next_state[i] = state[i]
                reward[i] = -1
            elif next_state[i] in other_state:
                for j in range(len(other_state)):
                    if next_state[i] == other_state[j] and state[i] == other_next_state[j]:
                        next_state[i] = state[i]
                        reward[i] = -1
        return reward, next_state

    def step(self, action_list):
        reward, next_state = self.get_reward(self.state, action_list)
        done = True
        for i in range(self.num_agent):
            if next_state[i] != self.goal[i]:
                done = False
        self.state = next_state
        return reward, done, self.state

    def sqr_dist(self, pos1, pos2):
        return (pos1[0]-pos2[0])*(pos1[0]-pos2[0])+(pos1[1]-pos2[1])*(pos1[1]-pos2[1])

    def get_global_obs(self):
        obs = np.zeros((self.high, self.width, 4))
        for i in range(self.high):
            for j in range(self.width):
                #if self.occupancy[i][j] == 0:
                    obs[i, j, 0] = 1.0
                    obs[i, j, 1] = 1.0
                    obs[i, j, 2] = 1.0
                    obs[i, j, 3] = 1.0
        for i in range(self.num_agent):
            if i%6 == 0:
                obs[self.state[i][0], self.state[i][1], 0] = 1.0
                obs[self.state[i][0], self.state[i][1], 1] = 0.0
                obs[self.state[i][0], self.state[i][1], 2] = 0.0
                obs[self.state[i][0], self.state[i][1], 3] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 0] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 1] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 2] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 3] = 0.0
            elif i%6 == 1:
                obs[self.state[i][0], self.state[i][1], 0] = 0.0
                obs[self.state[i][0], self.state[i][1], 1] = 1.0
                obs[self.state[i][0], self.state[i][1], 2] = 0.0
                obs[self.state[i][0], self.state[i][1], 3] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 0] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 1] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 2] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 3] = 0.0
            elif i%6 == 2:
                obs[self.state[i][0], self.state[i][1], 0] = 0.0
                obs[self.state[i][0], self.state[i][1], 1] = 0.0
                obs[self.state[i][0], self.state[i][1], 2] = 1.0
                obs[self.state[i][0], self.state[i][1], 3] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 0] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 1] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 2] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 3] = 1.0
            elif i%6 == 3:
                obs[self.state[i][0], self.state[i][1], 0] = 0.0
                obs[self.state[i][0], self.state[i][1], 1] = 0.0
                obs[self.state[i][0], self.state[i][1], 2] = 0.0
                obs[self.state[i][0], self.state[i][1], 3] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 0] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 1] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 2] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 3] = 1.0
            elif i%6 == 4:
                obs[self.state[i][0], self.state[i][1], 0] = 1.0
                obs[self.state[i][0], self.state[i][1], 1] = 0.0
                obs[self.state[i][0], self.state[i][1], 2] = 1.0
                obs[self.state[i][0], self.state[i][1], 3] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 0] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 1] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 2] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 3] = 0.0
            else:
                #第六组颜色
                obs[self.state[i][0], self.state[i][1], 0] = 0.0
                obs[self.state[i][0], self.state[i][1], 1] = 1.0
                obs[self.state[i][0], self.state[i][1], 2] = 0.0
                obs[self.state[i][0], self.state[i][1], 3] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 0] = 0.0
                obs[self.goal[i][0], self.goal[i][1], 1] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 2] = 1.0
                obs[self.goal[i][0], self.goal[i][1], 3] = 1.0

        return obs

    def plot_scene(self):
        plt.figure(figsize=(5, 5))
        plt.imshow(self.get_global_obs())
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def render(self):
        obs = self.get_global_obs()
        enlarge = 40
        henlarge = int(enlarge/2)
        qenlarge = int(enlarge/8)
        new_obs = np.ones((self.high*enlarge, self.width*enlarge, 3))
        for i in range(self.high):
            for j in range(self.width):
                if obs[i][j][0] == 0.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 0.0 and obs[i][j][3] == 0.0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (0, 0, 0), -1)
                if obs[i][j][0] == 1.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 0.0 and obs[i][j][3] == 0.0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (0, 0, 255), -1)
                if obs[i][j][0] == 1.0 and obs[i][j][1] == 1.0 and obs[i][j][2] == 0.0 and obs[i][j][3] == 0.0:
                    cv2.circle(new_obs, ((2*j+1) * henlarge, (2*i+1) * henlarge), henlarge, (0, 0, 255),-1)
                if obs[i][j][0] == 0.0 and obs[i][j][1] == 1.0 and obs[i][j][2] == 0.0 and obs[i][j][3] == 0.0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (0, 255, 0), -1)
                if obs[i][j][0] == 0.0 and obs[i][j][1] == 1.0 and obs[i][j][2] == 1.0 and obs[i][j][3] == 0.0:
                    cv2.circle(new_obs, ((2 * j + 1) * henlarge, (2 * i + 1) * henlarge), henlarge, (0, 255, 0), -1)
                if obs[i][j][0] == 0.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 1.0 and obs[i][j][3] == 0.0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (255, 0, 0), -1)
                if obs[i][j][0] == 0.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 1.0 and obs[i][j][3] == 1.0:
                    cv2.circle(new_obs, ((2 * j + 1) * henlarge, (2 * i + 1) * henlarge), henlarge, (255, 0, 0), -1)
                if obs[i][j][0] == 0.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 0.0 and obs[i][j][3] == 1.0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (255, 255, 0), -1)
                if obs[i][j][0] == 1.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 0.0 and obs[i][j][3] == 1.0:
                    cv2.circle(new_obs, ((2 * j + 1) * henlarge, (2 * i + 1) * henlarge), henlarge, (255, 255, 0), -1)
                if obs[i][j][0] == 1.0 and obs[i][j][1] == 0.0 and obs[i][j][2] == 1.0 and obs[i][j][3] == 0.0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (0, 255, 255), -1)
                if obs[i][j][0] == 1.0 and obs[i][j][1] == 1.0 and obs[i][j][2] == 1.0 and obs[i][j][3] == 0.0:
                    cv2.circle(new_obs, ((2 * j + 1) * henlarge, (2 * i + 1) * henlarge), henlarge, (0, 255, 255), -1)
                if obs[i][j][0] == 0.0 and obs[i][j][1] == 1.0 and obs[i][j][2] == 0.0 and obs[i][j][3] == 1.0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (255, 0, 255), -1)
                if obs[i][j][0] == 0.0 and obs[i][j][1] == 1.0 and obs[i][j][2] == 1.0 and obs[i][j][3] == 1.0:
                    cv2.circle(new_obs, ((2 * j + 1) * henlarge, (2 * i + 1) * henlarge), henlarge, (255, 0, 255), -1)
                if self.obstacles[i][j] == 1:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), (0, 0, 0), -1)
        cv2.imshow('image', new_obs)
        cv2.waitKey(600)