import copy
from env import CliffWalkingEnv

class PolicyIteration:
    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0] * (self.env.col * self.env.row) #初始化价值
        self.pi = [[0.25, 0.25, 0.25, 0.25]
                    for i in range(self.env.col*self.env.row)] #均匀随机策略
        self.theta = theta #收敛阈值
        self.gamma = gamma #折扣因子
        #print(self.env.P[0][0])

    def policy_evaluation(self):
        cnt = 1
        while 1:
            max_diff = 0
            new_v = [0] * (self.env.col * self.env.row)
            for s in range(self.env.col * self.env.row): #状态
                qsa_list = [] #Q(s,a)
                for a in range(4):
                    p, next_state, reward, done = self.env.P[s][a]
                    qsa = p * (reward + self.gamma * self.v[next_state] * (1-done))
                    qsa_list.append(self.pi[s][a] * qsa)
                new_v[s] = sum(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta : break #每次循环都把max_diff置零了
            cnt += 1
        print("策略评估经过%d轮后完成" % cnt)

    def policy_improvement(self):
        for s in range(self.env.row * self.env.col):
            qsa_list = []
            for a in range(4):
                p, next_state, reward, done = self.env.P[s][a]
                qsa = p * (reward + self.gamma * self.v[next_state] * (1-done))
                qsa_list.append(qsa)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq)
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]
        print("策略提升完成")
        return self.pi
    
    def policy_iteration(self):
        while 1:
            self.policy_evaluation()
            old_pi = copy.deepcopy(self.pi)
            new_pi = self.policy_improvement()
            if old_pi == new_pi: break
    
def print_agent(agent, action_meaning, disaster=[], ending=[]):
    print("状态价值：")
    for i in range(agent.env.row):
        for j in range(agent.env.col):
            print('%6.6s' % ('%.3f' % agent.v[i*agent.env.col+j]), end=' ')
        print()

    print("策略")
    for i in range(agent.env.row):
        for j in range(agent.env.col):
            if (i*agent.env.col+j) in disaster:
                print('****', end=' ')
            elif (i*agent.env.col+j) in ending:
                print('EEEE', end=' ')
            else:
                a = agent.pi[i*agent.env.col+j]
                pi_str=''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()

env = CliffWalkingEnv()
action_meaning = ['v','^','>','<']
theta = 0.001
gamma = 0.9
agent = PolicyIteration(env, theta, gamma)
agent.policy_iteration()
print_agent(agent, action_meaning, list(range(37,47)), [47])