import numpy as np

class Sarsa:
    def __init__(self, col, row, epsilon, alpha, gamma, n_action = 4):
        self.Q_table = np.zeros([col*row, n_action]) # (状态，动作)表格
        self.n_action = n_action
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon # 贪心中的参数
    
    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state):
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1, a1):
        td_error = r + self.gamma * self.Q_table[s1, a1] - self.Q_table[s0, a0]  #时序差分
        self.Q_table[s0, a0] += self. alpha * td_error
