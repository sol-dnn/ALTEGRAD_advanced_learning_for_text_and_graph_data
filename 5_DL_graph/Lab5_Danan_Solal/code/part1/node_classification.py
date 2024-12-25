"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye

from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score
from deepwalk import deepwalk


# Loads the karate network
G = nx.read_weighted_edgelist('/Users/dnn/M2DS 24-25/Courses/ALTEGRAD/Cour5_DL_graph/ALTEGRAD_lab_5_DLForGraphs_2024/Lab5_Danan_Solal/code/data/karate.edgelist', delimiter=' ', nodetype=int, create_using=nx.Graph())
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

n = G.number_of_nodes()

# Loads the class labels
class_labels = np.loadtxt('/Users/dnn/M2DS 24-25/Courses/ALTEGRAD/Cour5_DL_graph/ALTEGRAD_lab_5_DLForGraphs_2024/Lab5_Danan_Solal/code/data/karate_labels.txt', delimiter=',', dtype=np.int32)
idx_to_class_label = dict()
for i in range(class_labels.shape[0]):
    idx_to_class_label[class_labels[i,0]] = class_labels[i,1]

y = list()
for node in G.nodes():
    y.append(idx_to_class_label[node])

y = np.array(y)


############## Task 5
# Visualizes the karate network

##################
import matplotlib.pyplot as plt
color_map = ['red' if label == 0 else 'blue' for label in y] 
plt.figure(figsize=(10, 8))
nx.draw_networkx(
    G,
    node_color=color_map,
    with_labels=True,
    node_size=500,
    font_size=10,
    font_color='white'
)
plt.title("Karate Network Visualization", fontsize=20)
plt.show()
##################


############## Task 6
# Extracts a set of random walks from the karate network and feeds them to the Skipgram model
n_dim = 128
n_walks = 10
walk_length = 20
model = deepwalk(G, n_walks, walk_length, n_dim)

embeddings = np.zeros((n, n_dim))
for i, node in enumerate(G.nodes()):
    embeddings[i,:] = model.wv[str(node)]

idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[:int(0.8*n)]
idx_test = idx[int(0.8*n):]

X_train = embeddings[idx_train,:]
X_test = embeddings[idx_test,:]

y_train = y[idx_train]
y_test = y[idx_test]


############## Task 7
# Trains a logistic regression classifier and use it to make predictions


##################
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Classification Accuracy DeepWalk: {accuracy:.4f}")
##################


############## Task 8
# Generates spectral embeddings

##################
n = G.number_of_nodes()
A = nx.adjacency_matrix(G).toarray() 
D = np.diag([G.degree(node) for node in G.nodes()]) 
D_inv = np.linalg.inv(D)  
L_rw = np.eye(n) - np.dot(D_inv, A)  # Normalized random walk Laplacian: L_rw = I - D^(-1)A

# eigenvectors of the two smallest eigenvalues of L_rw
eigenvalues, eigenvectors = eigs(L_rw, k=2, which='SM')
spectral_embeddings = eigenvectors.real
X_train_spec = spectral_embeddings[idx_train, :]
X_test_spec = spectral_embeddings[idx_test, :] 


# logistic regression on spectral embeddings
clf_spec = LogisticRegression(max_iter=1000, random_state=42)
clf_spec.fit(X_train_spec, y_train)
y_pred_spec = clf_spec.predict(X_test_spec)
accuracy_spec = accuracy_score(y_test, y_pred_spec)

print(f"Classification Accuracy (Spectral Embeddings): {accuracy_spec:.4f}")
##################
