class CliffWalkingEnv:
    def __init__(self, col=12, row=4):
        self.col = col
        self.row = row
        # 状态转移矩阵P[state][action]=[(p, next_state, reward, done)]
        # p概率,next_state下一个位置,reward及时奖励,done是能否到达出口
        self.P = self.createP()
        
    def createP(self):
        P = [[[] for j in range(4)] for i in range(self.row * self.col)]
        change = [[0,1], [0, -1], [1,0], [-1,0]]
        for i in range(self.row):
            for j in range(self.col):
                for a in range(4):
                    if i == self.row-1 and j > 0: #处在最下面一行且不是起点的位置（悬崖）
                        P[i*self.col+j][a] = (1, i*self.col+j, 0, True)
                        continue
                    next_x = min(self.col-1, max(0, j+change[a][0]))
                    next_y = min(self.row-1, max(0, i+change[a][1]))
                    next_state = next_y * self.col + next_x
                    reward = -1
                    done = False
                    if next_y == self.row-1 and next_x > 0: #下一步是悬崖
                        done = True
                        if next_x != self.col-1:
                            reward = -100
                    P[i*self.col+j][a] = (1, next_state, reward, done)
        return P
