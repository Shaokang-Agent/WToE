import numpy as np
from ENV.env_two_rooms import EnvGoObstacle
from algorithm.MAXE_Q import MAXE_Q_learning

if __name__ == '__main__':
    max_episode = 500
    max_iteration = 10
    max_steps = 50
    map_size = 7
    rewards_agent = np.zeros([2, max_iteration, max_episode])
    for num_agent in [5, 10]:
        env = EnvGoObstacle(map_size, num_agent)
        Q_learning = [MAXE_Q_learning(map_size * map_size, 5) for _ in range(num_agent)]
        visited_num = np.zeros([map_size, map_size])
        for iter in range(max_iteration):
            for index in range(num_agent):
                Q = np.load("Q_value" + str(index % 2 + 1) + ".npy")
                Q_learning[index].set_Q_value(Q)
            all_steps = 0
            for i in range(max_episode):
                state = env.reset()
                count = 0
                done = False
                rewards = np.zeros(num_agent)
                while not done:
                    action_list = []
                    for index in range(num_agent):
                        s = state[index][0] * map_size + state[index][1]
                        visited_num[state[index][0]][state[index][1]] = visited_num[state[index][0]][state[index][1]] + 1
                        epsilon = np.min([0.9, 0.7+(0.9-0.7)*(i*max_steps+count)/(max_episode*max_steps/3)])
                        action, _, _ = Q_learning[index].choose_action(s, epsilon)
                        action_list.append(action)
                    reward, _, next_state = env.step(action_list)
                    if count > max_steps:
                        done = True
                    else:
                        count += 1
                    for index in range(num_agent):
                        s = state[index][0] * map_size + state[index][1]
                        a = action_list[index]
                        r = reward[index]
                        s_ = next_state[index][0]*map_size+next_state[index][1]
                        rewards[index] += r
                        Q_learning[index].learn(s,a,r,s_)
                    state = next_state
                rewards_agent[int(num_agent / 5 - 1)][iter][i] = np.mean(rewards)
                print(i, count)
                all_steps += count
            print(iter)
    np.save("maxeq_rewards.npy", rewards_agent)