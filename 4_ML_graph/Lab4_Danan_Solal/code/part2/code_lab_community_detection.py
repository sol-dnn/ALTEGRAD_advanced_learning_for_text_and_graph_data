"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans


############## Task 3
# Perform spectral clustering to partition graph G into k clusters


def spectral_clustering(G, k):
    ##################

    n = G.number_of_nodes()
    A = nx.adjacency_matrix(G)
    D = np.zeros((n,n))
    for i,node in enumerate(G.nodes()):
        D[i,i] = G.degree(node)
    D_inv = np.linalg.inv(D)                     # D^(-1)
    L_rw = np.eye(G.number_of_nodes()) - D_inv @ A  # L_rw = I - D^(-1)A
    eigenvalues, eigenvectors = eigs(L_rw, k=k, which='SR')
    eigenvectors = eigenvectors.real
    U = eigenvectors  #(n_nodes, k)
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(U)
    labels = kmeans.labels_
    clustering = {node: labels[i] for i, node in enumerate(G.nodes())}
    ##################

    return clustering

############## Task 4

##################
path = "/Users/dnn/M2DS 24-25/Courses/ALTEGRAD/Cour4_ML_graph/ALTEGRAD_lab_4_MLForGraphs_2024/Lab4_Danan_Solal/code/datasets/CA-HepTh.txt"  
G = nx.readwrite.edgelist.read_edgelist(path, delimiter='\t', comments="#", create_using=nx.Graph())
largest_cc_nodes = max(nx.connected_components(G), key=len)
gcc = G.subgraph(largest_cc_nodes)



# Apply Spectral Clustering to the GCC
num_clusters = 50
clustering = spectral_clustering(gcc, num_clusters)

#print(f"Cluster assignments for {len(clustering)} nodes:")
#for node, cluster in clustering.items():
    #print(f"Node {node}: Cluster {cluster}")

##################




############## Task 5

# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    
    ##################
    modularity = 0
    clusters = set(clustering.values())
    m = G.number_of_edges()  # Total number of edges in the graph

    for cluster in clusters:
        nodes_in_cluster = [node for node in G.nodes() if clustering[node] == cluster]
        subG = G.subgraph(nodes_in_cluster)
        l_c = subG.number_of_edges()
        d_c = sum(G.degree(node) for node in nodes_in_cluster)         # Sum of degrees of nodes in the cluster
        modularity += (l_c / m) - (d_c / (2 * m))**2

    ##################
    
    return modularity

'''
# To check our calculation by hands for question 2
G= nx.Graph()
edges = [
    (1, 2), (1, 3), (1, 4), (2, 3), (3, 4), (4, 5),  # Green cluster
    (6, 7), (6, 8), (6, 9), (7, 8), (7, 9), (8, 9),  # Blue cluster
    (5, 6),  # Between clusters
]
G.add_edges_from(edges)
clustering = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0}
mod = modularity(G, clustering)
print(f"Modularity for Figure 1(a): {mod}")

'''


############## Task 6

##################
print("Modularity spectral clustering:", modularity(gcc, clustering))

random_clustering = dict()
for node in gcc.nodes():
    random_clustering[node] = randint(0,49)
    
print("Modularity random clustering:", modularity(gcc, random_clustering))
##################


'''
Spectral Clustering Modularity:
The modularity value for spectral clustering (0.2202) is significantly higher than the random clustering modularity. 
This indicates that the spectral clustering algorithm has successfully grouped nodes into clusters with a strong community structure. 
Randomly assigning nodes to clusters generally results in partitions with no meaningful community structure. 
This explain why the modularity value for random clustering is very close to zero.
'''

