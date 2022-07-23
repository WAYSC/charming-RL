from asyncore import read
import copy
from tkinter import N
from typing_extensions import Self
from env import CliffWalkingEnv
from policy import print_agent

class ValueIteration:
    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0] * self.env.row * self.env.col
        self.theta = theta
        self.gamma = gamma
        self.pi = [None for i in range(self.env.row*self.env.col)] #价值迭代结束后的策略

    def value_iteration(self):
        cnt = 0
        while 1:
            max_diff = 0
            new_v = [0] * self.env.row * self.env.col
            for s in range(self.env.row * self.env.col):
                qsa_list = []
                for a in range(4):
                    p, nex_state, reward, done = self.env.P[s][a]
                    qsa = p * (reward + self.gamma * self.v[nex_state] * (1-done))
                    qsa_list.append(qsa)
                new_v[s] = max(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta: break
            cnt += 1
        print("价值迭代一共进行%d轮"%cnt)
        self.get_policy()
    
    def get_policy(self):
        for s in range(self.env.row*self.env.col):
            qsa_list = []
            for a in range(4):
                p, next_state, reward, done = self.env.P[s][a]
                qsa = reward + p * self.gamma * self.v[next_state] * (1-done)
                qsa_list.append(qsa)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq)
            self.pi[s] =  [1 / cntq if q == maxq else 0 for q in qsa_list]

env = CliffWalkingEnv()
action_meaning = ['v','^','>','<']
theta = 0.001
gamma = 0.9
agent = ValueIteration(env, theta, gamma)
agent.value_iteration()
print_agent(agent, action_meaning, list(range(37, 47)), [47])

