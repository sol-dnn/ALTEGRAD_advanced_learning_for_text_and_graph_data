"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1

path = "/Users/dnn/M2DS 24-25/Courses/ALTEGRAD/Cour4_ML_graph/ALTEGRAD_lab_4_MLForGraphs_2024/Lab4_Danan_Solal/code/datasets/CA-HepTh.txt"  

G = nx.readwrite.edgelist.read_edgelist(path, delimiter='\t', comments="#", create_using=nx.Graph())
print('Nodes: ', G.number_of_nodes())
print('Edges: ', G.number_of_edges())


############## Task 2

# Number of connected components
num_connected_components = nx.number_connected_components(G)
print('Number of connected components:', num_connected_components)

largest_cc_nodes = max(nx.connected_components(G), key=len)
largest_cc_subgraph = G.subgraph(largest_cc_nodes)

# Number of nodes and edges in the largest connected component
largest_cc_num_nodes = largest_cc_subgraph.number_of_nodes()
largest_cc_num_edges = largest_cc_subgraph.number_of_edges()
print('Number of nodes in the largest connected component:', largest_cc_num_nodes)
print('Number of edges in the largest connected component:', largest_cc_num_edges)

fraction_nodes = largest_cc_num_nodes / G.number_of_nodes()
fraction_edges = largest_cc_num_edges / G.number_of_edges()
print('Fraction of total nodes in the largest connected component:', fraction_nodes)
print('Fraction of total edges in the largest connected component:', fraction_edges)
