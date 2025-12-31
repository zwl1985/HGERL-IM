import networkx as nx
from tqdm import tqdm
import utils
import torch
import numpy as np
from models import NodeEncoder
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from environment import Environment
from agent import Agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_hyperedges_from_k_hop(g, k):
    # hs = np.empty((0, 0))
    # ht = np.empty((0, 0))
    list_of_source_hyperedges = []
    list_of_target_hyperedges = []

    for node in g.nodes():
        source_hyperedges = [0] * g.number_of_nodes()
        target_hyperedges = [0] * g.number_of_nodes()
        target_hyperedges[node] = 1
        a = {n: p for n, p in nx.single_source_shortest_path(g, node).items() if len(p)-1 == k}
        
        if len(a) == 0: 
            continue
        for dest, path in a.items():
            prob = 1
            current_node = path[0]
            for v in path[1:]:
                prob *= g.get_edge_data(current_node, v)['weight']
                current_node = v 
            source_hyperedges[dest] = prob
            source_hyperedges[node] = 0
        
        # hs = add_column(hs, source_hyperedges)
        # ht = add_column(ht, target_hyperedges)
        list_of_source_hyperedges.append(source_hyperedges)
        list_of_target_hyperedges.append(target_hyperedges)

    return list_of_source_hyperedges, list_of_target_hyperedges


def get_hyperedges_from_pair(g):
    list_of_source_hyperedges = []
    list_of_target_hyperedges = []

    for u, v, a in g.edges(data=True):
        source_hyperedges = [0] * g.number_of_nodes()
        target_hyperedges = [0] * g.number_of_nodes()
        source_hyperedges[v] = a['weight']
        target_hyperedges[u] = 1
        list_of_source_hyperedges.append(source_hyperedges)
        list_of_target_hyperedges.append(target_hyperedges)

    return list_of_source_hyperedges, list_of_target_hyperedges



def get_emb(g, hs, ht):
    nodes = torch.tensor([i for i in range(g.number_of_nodes())], dtype=torch.long, device=device)
    Ht = torch.tensor(ht, dtype=torch.float).to(device)
    Hs = torch.tensor(hs, dtype=torch.float).to(device)
    X = torch.ones((hs.shape[1], 128), dtype=torch.float, device=device)

 
    y_g = np.zeros(g.number_of_nodes(), dtype=float)
    for node in g.nodes():
        y_g[node] = utils.computeMC(g, [node], 10000) / g.number_of_nodes()

    y = y_g
    y = (y - y.min()) / (y.max() - y.min())

    y = torch.tensor(y, dtype=torch.float, device=device)       
        

    y.unsqueeze_(1)
    print(y)

    dataset = torch.utils.data.TensorDataset(nodes, y)

    data_loader = DataLoader(dataset, batch_size=16)

    model = NodeEncoder(128, hs.shape[0]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(200):
        model.train()
        train_loss = 0.0
        train_total = 0

        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
        
            outputs, x = model(X, X, Ht, Hs)
            out = outputs[batch_X]
            loss = F.mse_loss(out, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

           
            train_loss += loss.item() * batch_X.size(0)
            train_total += batch_y.size(0)

        
        train_loss = train_loss / train_total
        
        # print(f"Epoch {epoch+1}/{200}")
        # print(f"Train Loss: {train_loss:.4f}")
        # print("-" * 30)

    return x.cpu().detach().numpy()


def train(g, k, generation=2):


    hs_1, ht_1 = get_hyperedges_from_k_hop(g, 1)
    hs_pair, ht_pair = get_hyperedges_from_pair(g)

    hs = np.array(hs_1 + hs_pair,dtype=np.float32)
    ht = np.array(ht_1 + ht_pair, dtype=np.float32)

    

    x = get_emb(g, hs, ht)
    X = torch.tensor(x, dtype=torch.float32).to(device)

    env = Environment(g, k)
    Ht = torch.tensor(ht, dtype=torch.float).to(device)
    Hs = torch.tensor(hs, dtype=torch.float).to(device)
    agent = Agent(env, X, Hs, Ht, device)
    
    agent.initPop()
    

    for i in range(generation):  
        if i == 0:
            print("aaaa")
            agent.evaluate_all()
        best_train_fitness, average, rl_agent, elitePop = agent.train()
        print('Generation', i + 1, 'Epoch_Max:', '%.2f' % best_train_fitness,' Avg:', average, "influence:", best_train_fitness)


    influence = max(env.state_records.values())

    seed = next((k for k, v in env.state_records.items() if v == influence), None)
    print(f"seeds:{seed}  influence:{influence}")



if __name__ == '__main__':

    edge_list = []

    with open(r"C:\Users\Administrator\Desktop\HGERL-IM\graphs\netscience.txt", "r") as f:
        for line in f:
            u, v = line.strip().split()
            u = int(u)
            v = int(v)
            edge_list.append([u, v])
    g = nx.from_edgelist(edge_list, create_using=nx.DiGraph)

    for u, v, a in g.edges(data=True):
        a['weight'] = 1 / len(list(g.predecessors(v)))

    train(g, 10)