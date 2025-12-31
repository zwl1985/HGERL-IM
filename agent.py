import Replay_Memory
import Mod_Utils as Utils
from torch.optim import Adam
from Evolutionary_Algorithm import EA
import numpy as np
import torch
from torch import nn
import random
from models import DQN


class Agent:
    def __init__(self, env, X, Hs, Ht, device):
        self.is_cuda = True  # Set to use CUDA
        self.is_memory_cuda = True  # Set to use GPU graphics memory
        self.batch_size = 512  # Set batch size
        self.use_done_mask = True 
        self.pop_size = 100  # Set population size
        self.buffer_size = 10000  # Set cache pool size
        self.randomWalkTimes = 20  # Set the number of random point selections based on DQN
        self.learningTimes = 3  # Set the number of times to accelerate DQN training based on DRL technology 
        self.env = env  # Initialize the impact propagation environment
        self.evalStep = 1  # Set DQN point selection times based on the number of seed nodes
        self.evolver = EA(self.pop_size)  # initialization
        self.X = X
        self.Hs = Hs
        self.Ht = Ht        
        self.device = device

    def initPop(self):
        # Initialize DQN population
        self.pop = []
        for _ in range(self.pop_size):
            self.pop.append(DQN(self.X.shape[1], self.Hs.shape[0]).to(self.device))
        # Initialize the fitness array corresponding to the DQN population
        self.all_fitness = []
        # Turn off gradients and put in eval mode
        for dqn in self.pop:
            dqn.eval()
        # Initialize optimal DQN
        self.rl_agent = DQN(self.X.shape[1], self.Hs.shape[0]).to(self.device)
        # Initial DQN parameters
        self.gamma = 0.8  # Set update ratio
        self.optim = Adam(self.rl_agent.parameters(), lr=0.001)  # Set learners
        self.loss = nn.MSELoss()  # Set to use mean square error as the loss function
        self.replay_buffer = Replay_Memory.ReplayMemory(self.buffer_size)  # Initialize buffer pool
        # Initialize tracker parameters
        self.num_games = 0
        self.num_frames = 0
        self.gen_frames = 0

    # Store training quadruple data into a cache pool based on CUDA technology #
    def add_experience(self, state, action, next_state, reward, done):
        reward = Utils.to_tensor(np.array([reward])).unsqueeze(0)
        if self.is_cuda: reward = reward.cuda()
        if self.use_done_mask:
            done = Utils.to_tensor(np.array([done]).astype('uint8')).unsqueeze(0)
            if self.is_cuda: done = done.cuda()

        self.replay_buffer.push(state, action, next_state, reward, done)

    def take_action(self, state, net):
        selectable_nodes = list(set(self.env.graph.nodes()) - set(self.env.seeds))

        selectable_nodes_t = torch.tensor(selectable_nodes, dtype=torch.long, device=self.device)

        indices = torch.where(state == 1)[0]
        modified_Ht = self.Ht.clone()
        modified_Ht[:,indices] = 0

        # hs, ht = state2input(self.Hs.cpu().detach().numpy(), self.Ht.cpu().detach().numpy(), state.unsqueeze(1).cpu().detach().numpy())
        # Ht = torch.tensor(ht, dtype=torch.float32, device=device)
        # Hs = torch.tensor(hs, dtype=torch.float32, device=device)

        q_values = net.forward(
            self.X, self.X, modified_Ht, self.Hs, state.unsqueeze(1).to(self.device)
        ).squeeze(-1)
        # print(q_values)
        
        selectable_q_values_sort = q_values[selectable_nodes_t].sort(descending=True).values
        
        for mq in selectable_q_values_sort:
            max_position = set((q_values == mq).nonzero().view(-1).tolist())
            nodes = list(set(selectable_nodes).intersection(max_position))
            if len(nodes) > 0:
                node = random.choice(nodes)
                break
        return node

    # Select seed nodes based on the node score output by DQN and calculate the fitness value, while caching the quadruple data during the selection process #
    def evaluate(self, net):
        total_reward = 0.0
        state = self.env.reset()
        state = Utils.to_tensor(state).unsqueeze(0)
        if self.is_cuda:
            state = state.cuda()
        done = False

        while not done:
            actionInt = self.take_action(state.squeeze(0), net)  
            reward, next_state, done = self.env.step(actionInt)   
            action = torch.tensor([actionInt])
            next_state = Utils.to_tensor(next_state).unsqueeze(0)
            self.add_experience(state.cpu(), action, next_state.cpu(), reward, done)
            total_reward += reward
            state = next_state 
            
        return total_reward, self.env.seeds

    # Copy the network weights of the source DQN to the network weights of the target DQN #
    def rl_to_evo(self, rl_net, evo_net):
        for target_param, param in zip(evo_net.parameters(), rl_net.parameters()):
            target_param.data.copy_(param.data)

    # Evaluate the fitness value of the evolved DQN population #
    def evaluate_all(self):
        self.all_fitness = []
        for net in self.pop:
            fitness, _ = self.evaluate(net)
            self.all_fitness.append(fitness)
        print("all:", self.all_fitness)
        best_train_fitness = max(self.all_fitness)
        print("fitness_init:", best_train_fitness)

    # Training Evolutionary DQN Population Based on Evolutionary Algorithms and DRL Technology #
    def train(self):
        self.gen_frames = 0
        ####################### EVOLUTION #####################
        # Obtain optimal fitness value
        best_train_fitness = max(self.all_fitness)
        # Evolutionary algorithms are used to evolve the network weights of the DQN population and update the fitness values of the new population
        new_pop = self.evolver.epoch(self.pop, self.all_fitness)
        new_pop_fitness = []
        for net in new_pop:
            fitness, _ = self.evaluate(net)
            new_pop_fitness.append(fitness)
        self.pop, self.all_fitness = self.get_offspring(self.pop, self.all_fitness, new_pop, new_pop_fitness)
        
        print("epoch finished.    cost time:")
        # Obtain the optimal fitness value of the current DQN population
        fitness_best, _ = self.evaluate(self.pop[0])

        ####################### DRL Learning #####################
        # rl learning step
        
        # Based on the n-step Q-learning technique in DRL thinking, the optimal DQN is reverse updated using empirical data from the cache pool, and its network weights are copied to the DQN with poor fitness values
        for _ in range(self.learningTimes):
            index = random.randint(len(self.pop) // 2, len(self.pop) - 1)
            self.rl_to_evo(self.pop[0], self.rl_agent)
            if len(self.replay_buffer) > self.batch_size * 2:
                transitions = self.replay_buffer.sample(self.batch_size)
                batch = Replay_Memory.Transition(*zip(*transitions))
                self.update_parameters(batch)
                fitness, _ = self.evaluate(self.rl_agent)
                if fitness_best < fitness:
                    self.rl_to_evo(self.rl_agent, self.pop[index])
                    self.all_fitness[index] = fitness

        
        print("learning finished.")
        return best_train_fitness, sum(self.all_fitness) / len(self.all_fitness), self.rl_agent, self.pop[
                                                                                                 0:len(self.pop) // 10]

    # Based on the n-step Q-learning technique in DRL thinking, error values are calculated using empirical data of specific batch sizes in the cache pool, and the network weights of DQN are updated in reverse using error value gradients using random gradient descent technique #
    def update_parameters(self, batch):
        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        done_batch = None
        if self.use_done_mask: done_batch = torch.cat(batch.done)

        state_batch.requires_grad = False
        next_state_batch.requires_grad = True
        action_batch.requires_grad = False

        # Load everything to GPU if not already
        if self.is_cuda:
            self.rl_agent.cuda()
            state_batch = state_batch.cuda()
            next_state_batch = next_state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            if self.use_done_mask: done_batch = done_batch.cuda()

        currentList = torch.Tensor([])
        currentList = torch.unsqueeze(currentList, 1).cuda()
        targetList = torch.Tensor([])
        targetList = torch.unsqueeze(targetList, 1).cuda()
        # DQN Update
        for state, action, reward, next_state, done in zip(state_batch, action_batch, reward_batch, next_state_batch,
                                                           done_batch):
            target = torch.Tensor([reward])
            if not done:

                indices = torch.where(next_state== 1)[0]
                modified_Ht = self.Ht.clone()
                modified_Ht[:,indices] = 0
                
                # hs, ht = state2input(self.Hs.cpu().detach().numpy(), self.Ht.cpu().detach().numpy(), next_state.unsqueeze(1).cpu().detach().numpy())
                # hs, ht = state2input(self.Hs.detach().numpy(), self.Ht.detach().numpy(), next_state.unsqueeze(1).detach().numpy())
                # Ht = torch.tensor(ht, dtype=torch.float32, device=device)
                # Hs = torch.tensor(hs, dtype=torch.float32, device=device)

                next_q_values = self.rl_agent.forward(
            self.X, self.X, modified_Ht, self.Hs, next_state.unsqueeze(1).to(self.device)
        )


                # next_q_values = self.rl_agent.forward(next_state)
                pred, idx = next_q_values.max(0)
                target = reward + self.gamma * pred


            indices = torch.where(state == 1)[0]
            modified_Ht = self.Ht.clone()
            modified_Ht[:,indices] = 0
            
            # hs, ht = state2input(self.Hs.cpu().detach().numpy(), self.Ht.cpu().detach().numpy(), state.unsqueeze(1).cpu().detach().numpy())
            # Ht = torch.tensor(ht, dtype=torch.float32, device=device)
            # Hs = torch.tensor(hs, dtype=torch.float32, device=device)


            target_f = self.rl_agent.forward(
            self.X, self.X, modified_Ht, self.Hs, state.unsqueeze(1).to(self.device)
        )


            # target_f = self.rl_agent.forward(state)

            current = target_f[action]
            current = torch.unsqueeze(current, 1)
            target = torch.unsqueeze(target, 1).cuda()
            currentList = torch.cat((currentList, current), 0)
            targetList = torch.cat((targetList, target), 0)

        self.optim.zero_grad()
        dt = self.loss(currentList, targetList)
        dt.backward()
        nn.utils.clip_grad_norm_(self.rl_agent.parameters(), 10000)
        self.optim.step()

        # Nets back to CPU if using memory_cuda
        if self.is_memory_cuda and not self.is_cuda:
            self.rl_agent.cpu()

    # In the DQN population that has evolved beyond the population size limit, first sort all DQN populations, then select the top 50 DQN populations to be retained, and randomly select 50 populations from the remaining populations to be retained #
    def get_offspring(self, pop, fitness_evals, new_pop, new_fitness_evals):
        all_pop = []
        fitness = []
        offspring = []
        offspring_fitness = []
        for i in range(len(pop)): 
            all_pop.append(pop[i])
            fitness.append(fitness_evals[i])
        for i in range(len(new_pop)): 
            all_pop.append(new_pop[i])
            fitness.append(new_fitness_evals[i])

        index_rank = sorted(range(len(fitness)), key=fitness.__getitem__) 
        index_rank.reverse()
        for i in range(len(pop) // 2): 
            offspring.append(all_pop[index_rank[i]])
            offspring_fitness.append(fitness[index_rank[i]])

        randomNum = len(all_pop) - len(pop) // 2
        randomList = list(range(randomNum))
        random.shuffle(randomList)
        for i in range(len(pop) // 2, len(pop)):
            index = randomList[i - len(pop) // 2]
            offspring.append(all_pop[index])
            offspring_fitness.append(fitness[index])
            ...

        return offspring, offspring_fitness

    # Format output fitness score #
    def showScore(self, score):
        out = ""
        for i in range(len(score)):
            out = out + str(score[i])
            out = out + "\t"
        print(out)