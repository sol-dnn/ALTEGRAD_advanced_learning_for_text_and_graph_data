"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import numpy as np
import networkx as nx
from random import randint
from gensim.models import Word2Vec


############## Task 1
# Simulates a random walk of length "walk_length" starting from node "node"

def random_walk(G, node, walk_length):
    ##################
    walk = [node]
    for i in range(walk_length - 1):
        neighbors = list(G.neighbors(walk[-1]))
        if len(neighbors) > 0:
            walk.append(neighbors[randint(0, len(neighbors) - 1)])
        else:
            break
    ##################
    walk = [str(node) for node in walk]

    return walk



############## Task 2
# Runs "num_walks" random walks from each node
def generate_walks(G, num_walks, walk_length):
    walks = []
    
    ##################
    for _ in range(num_walks):
        for node in G.nodes():
            walks.append(random_walk(G, node, walk_length))
    np.random.shuffle(walks)  # Shuffle to remove any structure in the list of walks
    permuted_walks = walks
    ##################

    return permuted_walks


# Simulates walks and uses the Skipgram model to learn node representations
def deepwalk(G, num_walks, walk_length, n_dim):
    print("Generating walks")
    walks = generate_walks(G, num_walks, walk_length)

    print("Training word2vec")
    model = Word2Vec(vector_size=n_dim, window=8, min_count=0, sg=1, workers=8, hs=1)
    model.build_vocab(walks)
    model.train(walks, total_examples=model.corpus_count, epochs=5)

    return model
