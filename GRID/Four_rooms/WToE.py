import numpy as np
import sys
sys.path.append("../")
from ENV.env_four_rooms import EnvGoObstacle
from algorithm.Q_learning import Q_learning
from algorithm.WToE_Q_learning import WToE_Q_learning
from algorithm.WToE_repaly_buffer import WToE_Buffer
import argparse

parser = argparse.ArgumentParser("Reinforcement Learning experiments for Grid environments")
parser.add_argument("--lr-vae", type=float, default=1e-3, help="learning rate of vae")
parser.add_argument("--buffer_size", type=int, default=int(500), help="number of transitions can be stored in buffer")
parser.add_argument("--batch_size", type=int, default=32, help="number of episodes to optimize at the same time")
parser.add_argument("--history_size", type=int, default=15, help="number of history transitions to optimize at the same time")

parser.add_argument("--layers_before_gru", nargs='+', type=int, default=[])
parser.add_argument("--layers_after_gru", nargs='+', type=int, default=[])
parser.add_argument("--gru_hidden_size", type=int, default = 64)
parser.add_argument("--latent_dim", type = int, default = 32)
parser.add_argument("--state_embedding_size", type=int, default=10)
parser.add_argument("--action_embedding_size", type=int, default=10)
parser.add_argument("--reward_embedding_size", type=int, default=5)
parser.add_argument("--decoder_layers", nargs='+', type=int, default=[32])

parser.add_argument("--high", type=int, default=11)
parser.add_argument("--width", type=int, default=5)

args = parser.parse_args()


if __name__ == '__main__':
    single = False
    if single == True:
        for i in range(12):
            env = EnvGoObstacle(single, i)
            max_episode = 500
            max_iteration = 1
            max_steps = 50
            Q_learning_i = Q_learning(11 * 5, 5)
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
        num_agent = 12
        env = EnvGoObstacle(single, -1)
        max_episode = 500
        max_iteration = 5
        max_steps = 100
        VBAE = [WToE_Q_learning(args, 11 * 5, 5) for _ in range(num_agent)]
        Repaly_buffer = [WToE_Buffer(args) for _ in range(num_agent)]
        visited_num = np.zeros([max_episode, 11, 5])
        policy_differ_num = np.zeros([max_episode, 11, 5])
        z_num = np.zeros([max_episode, 11, 5])
        Rewards = np.zeros([max_iteration, max_episode])
        steps_mean = 0
        for iter in range(max_iteration):
            for index in range(num_agent):
                Q = np.load("Q_value_" + str(index) + ".npy")
                VBAE[index].set_Q_value(Q)
            all_steps = 0
            for i in range(max_episode):
                state = env.reset()
                count = 0
                done = False
                rewards = np.zeros(num_agent)
                state_stack = [[] for _ in range(num_agent)]
                action_stack = [[] for _ in range(num_agent)]
                reward_stack = [[] for _ in range(num_agent)]
                while not done:
                    action_list = []
                    for index in range(num_agent):
                        s = state[index][0] * 5 + state[index][1]
                        visited_num[i][state[index][0]][state[index][1]] = visited_num[i][state[index][0]][
                                                                               state[index][1]] + 1
                        if count == 0:
                            for h in range(args.history_size + 2):
                                state_stack[index].append(s)
                        else:
                            state_stack[index].pop(0)
                            state_stack[index].append(s)

                        epsilon = np.min(
                            [0.9, 0.7 + (0.9 - 0.7) * (i * max_steps + count) / (max_episode * max_steps / 3)])
                        action, flag, z_norm = VBAE[index].choose_action(s, epsilon, count, state_stack[index],
                                                                 action_stack[index], reward_stack[index])
                        z_num[i][state[index][0]][state[index][1]] = (z_num[i][state[index][0]][state[index][1]] * (
                                    visited_num[i][state[index][0]][state[index][1]] - 1) + z_norm) / \
                                                                     visited_num[i][state[index][0]][state[index][1]]
                        action_list.append(action)
                        if flag == 1:
                            policy_differ_num[i][state[index][0]][state[index][1]] += 1

                        if count == 0:
                            for h in range(args.history_size + 2):
                                action_stack[index].append(action)
                        else:
                            action_stack[index].pop(0)
                            action_stack[index].append(action)

                    reward, _, next_state = env.step(action_list)
                    for index in range(num_agent):
                        if count == 0:
                            for h in range(args.history_size + 2):
                                reward_stack[index].append(reward[index])
                        else:
                            reward_stack[index].pop(0)
                            reward_stack[index].append(reward[index])

                    if count > max_steps:
                        done = True
                    else:
                        count += 1

                    for index in range(num_agent):
                        s = state[index][0] * 5 + state[index][1]
                        a = action_list[index]
                        r = reward[index]
                        s_ = next_state[index][0] * 5 + next_state[index][1]
                        rewards[index] += r
                        Repaly_buffer[index].store_episode(s, a, r, s_)
                        if count > 32 or i > 1:
                            VBAE[index].learn(s, a, r, s_, Repaly_buffer[index])
                    state = next_state
                Rewards[iter][i] = np.mean(rewards)
                all_steps += count
                print(i, count)
            steps_mean += all_steps
            print(iter)
        print(steps_mean / max_iteration)
        np.save("rewards_WToE.npy", Rewards)