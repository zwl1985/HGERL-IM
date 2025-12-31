import utils
import numpy as np
import multiprocessing
import statistics

class Environment:
    def __init__(self, graph, k, gamma=0.99, n_steps=1, method='MC', R=10000, num_workers=5):
        """
        G: networkx的图，Graph或DiGraph；
        k: 种子集大小；
        n_steps: 计算奖励时的步长；
        method: 计算奖励的方法；
        R: 使用蒙特卡洛估计奖励的轮数；
        num_workers: 使用多少个核心计算传播范围
        """
        # self.graphs = graphs  # 子图列表
        self.k = k
        self.gamma = gamma
        self.n_steps = n_steps
        self.method = method
        self.R = R
        self.num_workers = num_workers
        self.graph = graph  # 当前使用的子图
        # 当前状态，每个位置表示一个节点是否被选择，1是已选，0是未选
        self.state = None
        # 前一状态的奖励
        self.preview_reward = 0.0
        # 记录每次探索的状态、动作、奖励、下一步状态，以便计算n步奖励（为了学习得更好，n步可以更好反应真实的情况）
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        
        self.seeds = []
        self.state_records = {}  # 记录种子集的奖励

    def reset(self):
        """
        重置环境。
        """
        # self.graph = random.choice(self.graphs)  # 随机选一个子图
        self.seeds = []
        self.state = np.zeros(self.graph.number_of_nodes(), dtype=np.float32)
        self.preview_reward = 0.0
        self.states = []
        self.actions = []
        self.rewards = []
        # self.next_states = []
        return self.state

    def step(self, action):
        """
        根据所给的action，转移到新状态。
        """
        
        self.states.append(self.state.copy())
        self.state[action] = 1   # 更新状态
        self.seeds.append(action)
        # 计算奖励
        reward = self.compute_reward()

        done = False
        if len(self.seeds) == self.k:
            done = True

        if done:
            self.states.append(self.state.copy())

        self.actions.append(action)
        self.rewards.append(reward)
        # self.next_states.append(self.state)
        return reward, self.state, done

    def compute_reward(self):
        """
        计算奖励，可以使用MC的方法或其他方法。
        """
        str_seeds = str(sorted(self.seeds))
        if self.method == 'MC':
            if str_seeds in self.state_records:
                current_reward = self.state_records[str_seeds]
            else:
                with multiprocessing.Pool(self.num_workers) as pool:
                    args = [[self.graph, self.seeds, int(self.R / self.num_workers)] for _ in
                            range(self.num_workers)]
                    results = pool.starmap(utils.computeMC, args)
                current_reward = statistics.mean(results)
            r = current_reward - self.preview_reward
            self.preview_reward = current_reward
            self.state_records[str_seeds] = current_reward
            return r
        else:
            pass