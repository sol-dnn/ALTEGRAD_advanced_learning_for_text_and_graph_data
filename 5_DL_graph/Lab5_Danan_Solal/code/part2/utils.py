"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import scipy.sparse as sp
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

def normalize_adjacency(A):
    ############## Task 9

    ##################
    n = A.shape[0]
    A_tilde = A + sp.identity(n)

    D_tilde_diag = np.array(A_tilde.sum(axis=1)).flatten()
    D_tilde_inv_sqrt = 1.0 / np.sqrt(D_tilde_diag)
    D_tilde_inv_sqrt[np.isinf(D_tilde_inv_sqrt)] = 0 
    D_tilde_inv_sqrt = sp.diags(D_tilde_inv_sqrt)

    A_normalized = D_tilde_inv_sqrt @ A_tilde @ D_tilde_inv_sqrt

    ##################

    return A_normalized


def load_cora():
    idx_features_labels = np.genfromtxt("/Users/dnn/M2DS 24-25/Courses/ALTEGRAD/Cour5_DL_graph/ALTEGRAD_lab_5_DLForGraphs_2024/Lab5_Danan_Solal/code/data/cora.content", dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    features = features.todense()
    features /= features.sum(1).reshape(-1, 1)
    
    class_labels = idx_features_labels[:, -1]
    le = LabelEncoder()
    class_labels = le.fit_transform(class_labels)

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("/Users/dnn/M2DS 24-25/Courses/ALTEGRAD/Cour5_DL_graph/ALTEGRAD_lab_5_DLForGraphs_2024/Lab5_Danan_Solal/code/data/cora.cites", dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(class_labels.size, class_labels.size), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))

    return features, adj, class_labels


def sparse_to_torch_sparse(M):
    """Converts a sparse SciPy matrix to a sparse PyTorch tensor"""
    M = M.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((M.row, M.col)).astype(np.int64))
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
