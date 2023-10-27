import numpy as np
class EITI_Q_learning():
    """docstring for DQN"""
    def __init__(self, state_num, action_num):
        super(EITI_Q_learning, self).__init__()
        self.action_num = action_num
        self.Q_value = np.zeros([state_num, action_num])

    def choose_action(self, state, episolon):
        if np.random.rand() <= episolon:
            action_value = self.Q_value[state]
            max_action = np.where(action_value == np.max(action_value))
            action = max_action[0][np.random.randint(0, len(max_action))]
        else:
            action = np.random.randint(0, self.action_num)
        return action

    def learn(self, state, action, reward, next_state, intrisic_reward):
        self.Q_value[state, action] = reward + 0.1 * intrisic_reward + 0.99 * np.max(self.Q_value[next_state])

    def get_Q_value(self):
        return self.Q_value

    def set_Q_value(self, Q):
        self.Q_value = Q