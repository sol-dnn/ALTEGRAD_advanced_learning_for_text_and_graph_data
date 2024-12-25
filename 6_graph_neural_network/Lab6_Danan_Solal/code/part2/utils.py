"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
import torch
from random import randint

def create_dataset():
    Gs = list()
    y = list()

    ############## Task 5
    
    ##################
    for _ in range(50):
        # Class 0: G(n, 0.2) model
        n = randint(10, 20) 
        G = nx.fast_gnp_random_graph(n, 0.2)  # Generate a random graph with p=0.2
        Gs.append(G)
        y.append(0) 

        # Class 1: G(n, 0.4) model
        n = randint(10, 20) 
        G = nx.fast_gnp_random_graph(n, 0.4) 
        Gs.append(G) 
        y.append(1)
    ##################

    return Gs, y


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
