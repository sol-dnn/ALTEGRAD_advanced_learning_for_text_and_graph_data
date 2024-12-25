"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN(nn.Module):
    """Simple GNN model"""
    def __init__(self, n_feat, n_hidden_1, n_hidden_2, n_class, dropout):
        super(GNN, self).__init__()

        self.fc1 = nn.Linear(n_feat, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_class)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x_in, adj):
        ############## Tasks 10 and 13
        
        ##################
        # First message passing layer: Z0 = f(Â X W0)
        z0 = self.fc1(x_in)   
        z0 = torch.mm(adj, z0)                                      
        z0 = self.relu(z0)                       
        z0 = self.dropout(z0)            

        # Second message passing layer: Z1 = f(Â Z0 W1)
        z1 = self.fc2(z0)   
        z1 = torch.mm(adj, z1)                                      
        z1 = self.relu(z1)                        
        z1 = self.dropout(z1)                     

        # Fully connected layer: Ŷ = softmax(Z1 W2)
        z2 = self.fc3(z1)                          
        ##################


        return F.log_softmax(z2, dim=1), z1
