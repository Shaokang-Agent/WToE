import numpy as np
import sys
sys.path.append("./")
from ENV.env_four_rooms import EnvGoObstacle
from algorithm.EITI_Q import EITI_Q_learning
import math

if __name__ == '__main__':
    single = False
    if single == True:
        for i in range(12):
            env = EnvGoObstacle(single, i)
            max_episode = 500
            max_iteration = 1
            max_steps = 50
            Q_learning_i = EITI_Q_learning(11 * 5, 5)
            for iter in range(max_iteration):
                Q = np.zeros([11 * 5, 5])
                Q_learning_i.set_Q_value(Q)
                all_steps = 0
                for j in range(max_episode):
                    state = env.reset()[0]
                    count = 0
                    done = False
                    rewards = 0
                    while not done:
                        s = state[0] * 5 + state[1]
                        epsilon = np.min([0.999, 0.7+(0.999-0.7)*(i*max_steps+count)/(max_episode*max_steps/3)])
                        a = Q_learning_i.choose_action(s, epsilon)
                        r, d, next_state = env.step([a])
                        next_state = next_state[0]
                        if count > max_steps:
                            done = True
                        else:
                            count += 1
                        s_ = next_state[0]*5+next_state[1]
                        Q_learning_i.learn(s,a,r,s_)
                        state = next_state
                    print(j, count)
                    all_steps += count
            np.save("Q_value_" + str(i)+ ".npy", Q_learning_i.get_Q_value())
    else:
        env = EnvGoObstacle(single, -1)
        max_episode = 500
        max_iteration = 10
        max_steps = 100
        steps_mean = 0
        Q_learning = [EITI_Q_learning(11 * 5, 5) for _ in range(12)]
        visited_num_sa = np.zeros(
            [12, 11 * 5, 5, 12, 11 * 5, 5, 11 * 5])
        visited_num = np.zeros([11, 5])
        Rewards = np.zeros([max_iteration, max_episode])
        for iter in range(max_iteration):
            for index in range(12):
                Q = np.load("Q_value_" + str(index) + ".npy")
                Q_learning[index].set_Q_value(Q)
            all_steps = 0
            for i in range(max_episode):
                state = env.reset()
                count = 0
                done = False
                rewards = np.zeros(12)
                while not done:
                    action_list = []
                    for index in range(12):
                        s = state[index][0] * 5 + state[index][1]
                        visited_num[state[index][0]][state[index][1]] = visited_num[state[index][0]][
                                                                            state[index][1]] + 1
                        epsilon = np.min(
                            [0.9, 0.7 + (0.9 - 0.7) * (i * max_steps + count) / (max_episode * max_steps / 3)])
                        action = Q_learning[index].choose_action(s, epsilon)
                        action_list.append(action)
                    reward, _, next_state = env.step(action_list)
                    for index in range(12):
                        s = state[index][0] * 5 + state[index][1]
                        a = action_list[index]
                        for other_index in range(12):
                            if other_index != index:
                                s_other = state[other_index][0] * 5 + state[other_index][1]
                                a_other = action_list[other_index]
                                s_next_other = next_state[other_index][0] * 5 + next_state[other_index][1]
                                visited_num_sa[index, s, a, other_index, s_other, a_other, s_next_other] += 1
                    if count > max_steps:
                        done = True
                    else:
                        count += 1
                    for index in range(12):
                        s = state[index][0] * 5 + state[index][1]
                        a = action_list[index]
                        r = reward[index]
                        s_ = next_state[index][0] * 5 + next_state[index][1]
                        rewards[index] += r
                        r_in = 0
                        for other_index in range(12):
                            if other_index != index:
                                s_other = state[other_index][0] * 5 + state[other_index][1]
                                a_other = action_list[other_index]
                                s_next_other = next_state[other_index][0] * 5 + next_state[other_index][1]
                                if np.sum(visited_num_sa[index, s, a, other_index, s_other, a_other, :]) == 0 or np.sum(
                                        visited_num_sa[index, :, :, other_index, s_other, a_other, :]) == 0:
                                    r_temp = 0
                                else:
                                    r_temp = math.log((visited_num_sa[
                                                           index, s, a, other_index, s_other, a_other, s_next_other] / np.sum(
                                        visited_num_sa[index, s, a, other_index, s_other, a_other, :])) /
                                                      (np.sum(visited_num_sa[index, :, :, other_index, s_other, a_other,
                                                              s_next_other]) / np.sum(
                                                          visited_num_sa[index, :, :, other_index, s_other, a_other,
                                                          :])))
                                r_in += r_temp
                        Q_learning[index].learn(s, a, r, s_, r_in)
                    state = next_state
                Rewards[iter][i] = np.mean(rewards)
                print(i, count)
                all_steps += count
            steps_mean += all_steps
            print(iter)
        print(steps_mean / max_iteration)
        np.save("rewards_EITI.npy", Rewards)