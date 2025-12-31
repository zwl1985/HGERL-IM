import torch
from torch import nn
import torch.nn.functional as F

class HypergraphConv(nn.Module):
    def __init__(self, in_feat, out_feat, num_edge, bias=True):
        super().__init__()
        self.weight_trans = nn.Parameter(torch.Tensor(in_feat, out_feat), requires_grad=True)
        nn.init.normal_(self.weight_trans, mean=0, std=0.01)
        self.edge_weight = nn.Parameter(torch.ones(num_edge), requires_grad=True)
        self.theta = nn.Parameter(torch.Tensor(in_feat, out_feat), requires_grad=True)
        nn.init.normal_(self.theta, mean=0, std=0.01)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat), requires_grad=True)
            nn.init.normal_(self.bias, mean=0, std=0.01)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, Xi, X, Ht, Hs):
        Xt = Ht.T @ torch.diag(self.edge_weight) @ Hs @ X @ self.theta + Xi @ self.weight_trans
        if self.bias is not None:
            Xt += self.bias
        return F.leaky_relu(Xt)
    

class NodeEncoder(nn.Module):
    def __init__(self, in_feat, num_edge, n_hid=[128, 128, 128], dropout = 0.5):
        super().__init__()
        self.conv0 = HypergraphConv(in_feat, n_hid[0], num_edge)
        self.conv1 = HypergraphConv(n_hid[0], n_hid[1], num_edge)
        self.conv1 = HypergraphConv(n_hid[1], n_hid[2], num_edge)
        self.dropout = dropout
        self.fc1 = nn.Linear(n_hid[-1], 128)
        self.fc2 = nn.Linear(128, 1)
    def forward(self, Xi, X, Ht, Hs):
        X0 = self.conv0(Xi, X, Ht, Hs)
        X0 = F.dropout(X0, self.dropout)
        X1 = self.conv1(Xi, X0, Ht, Hs)
        X1 = F.dropout(X1, self.dropout)
        X2 = self.conv1(Xi, X1, Ht, Hs)
        X3 = self.fc1(X2)
        return self.fc2(X3) ,X3
    

class DQN(nn.Module):
    def __init__(self, in_feat, num_edge, n_hid=[128, 128]):
        super().__init__()
        self.conv0 = HypergraphConv(in_feat, n_hid[0], num_edge)
        self.conv1 = HypergraphConv(n_hid[0], n_hid[1], num_edge)
        # self.conv2 = HypergraphConv(n_hid[1], n_hid[2], num_edge)
        self.fc = nn.Linear(n_hid[-1] + 1, 1)
    def forward(self, xi, x, Ht, Hs, state):
        # X = torch.cat([x, state], dim=1)
        X1 = self.conv0(xi, x, Ht, Hs)
        X1 = F.dropout(X1, p=0.5)
        X2 = self.conv1(xi, X1, Ht, Hs)
        X = F.leaky_relu(X2)
        X = self.fc(torch.cat([X, state], dim=1))
        
        return X