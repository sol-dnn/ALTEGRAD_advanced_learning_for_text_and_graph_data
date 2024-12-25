"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    """GAT layer"""
    def __init__(self, n_feat, n_hidden, alpha=0.05):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(n_feat, n_hidden, bias=False)
        self.a = nn.Linear(2*n_hidden, 1)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, adj):
        
        ############## Task 1
    
        ##################
        h_transformed = self.fc(x)  # Shape: [N, n_hidden]
        indices = adj.coalesce().indices()  # Shape is :[2, number of edges]
        src, dst = indices[0], indices[1]  # Separate source and destination node indices
        h_src = h_transformed[src]  # [E, n_hidden]
        h_dst = h_transformed[dst]
        edge_features = torch.cat([h_src, h_dst], dim=1)  # [E, 2 * n_hidden]
        # Compute attention scores
        h = self.leakyrelu(self.a(edge_features)).squeeze()  # [E]

        ##################

        h = torch.exp(h.squeeze())
        unique = torch.unique(indices[0,:])
        t = torch.zeros(unique.size(0), device=x.device)
        h_sum = t.scatter_add(0, indices[0,:], h)
        h_norm = torch.gather(h_sum, 0, indices[0,:])
        alpha = torch.div(h, h_norm)
        adj_att = torch.sparse.FloatTensor(indices, alpha, torch.Size([x.size(0), x.size(0)])).to(x.device)  #Compute the attention-weighted adjacency matrix
        
        
        ##################
        # Perform message passing (A ⊙ T) * H * W
        out = torch.sparse.mm(adj_att, h_transformed)  # Shape: [N, n_hidden]
        ##################

        return out, alpha



class GNN(nn.Module):
    """GNN model"""
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GNN, self).__init__()
        self.mp1 = GATLayer(nfeat, nhid)
        self.mp2 = GATLayer(nhid, nhid)
        self.fc = nn.Linear(nhid, nclass)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        
        ############## Tasks 2 and 4
    
        ##################
        # First message passing layer
        x, _ = self.mp1(x, adj)
        x = self.relu(x)
        x = self.dropout(x) 
        # Second message passing layer
        x, attention_scores = self.mp2(x, adj) 
        x = self.relu(x)  
        # Fully connected 
        x = self.fc(x) 
        ##################

        return F.log_softmax(x, dim=1), attention_scores

