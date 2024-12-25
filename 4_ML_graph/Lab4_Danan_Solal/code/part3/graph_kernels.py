import numpy as np
import re
from nltk.stem.porter import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

def load_file(filename):
    labels = []
    docs =[]

    with open(filename, encoding='utf8', errors='ignore') as f:
        for line in f:
            content = line.split(':')
            labels.append(content[0])
            docs.append(content[1][:-1])
    
    return docs,labels  


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower().split()


def preprocessing(docs): 
    preprocessed_docs = []
    n_sentences = 0
    stemmer = PorterStemmer()

    for doc in docs:
        clean_doc = clean_str(doc)
        preprocessed_docs.append([stemmer.stem(w) for w in clean_doc])
    
    return preprocessed_docs
    
    
def get_vocab(train_docs, test_docs):
    vocab = dict()
    
    for doc in train_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)

    for doc in test_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)
        
    return vocab


path_to_train_set = '/Users/dnn/M2DS 24-25/Courses/ALTEGRAD/Cour4_ML_graph/ALTEGRAD_lab_4_MLForGraphs_2024/Lab4_Danan_Solal/code/datasets/train_5500_coarse.label'
path_to_test_set = '/Users/dnn/M2DS 24-25/Courses/ALTEGRAD/Cour4_ML_graph/ALTEGRAD_lab_4_MLForGraphs_2024/Lab4_Danan_Solal/code/datasets/TREC_10_coarse.label'

# Read and pre-process train data
train_data, y_train = load_file(path_to_train_set)
train_data = preprocessing(train_data)

# Read and pre-process test data
test_data, y_test = load_file(path_to_test_set)
test_data = preprocessing(test_data)

# Extract vocabulary
vocab = get_vocab(train_data, test_data)
print("Vocabulary size: ", len(vocab))


import networkx as nx
import matplotlib.pyplot as plt

# Task 11

def create_graphs_of_words(docs, vocab, window_size):
    graphs = list()
    for idx,doc in enumerate(docs):
        G = nx.Graph()
        ##################
        
        # Add nodes to the graph for each unique word in the document
        for word in doc:
            if word in vocab:
                G.add_node(vocab[word], label=word)
        for i in range(len(doc)):
            for j in range(i + 1, min(i + window_size, len(doc))):
                if doc[i] in vocab and doc[j] in vocab:
                    node1 = vocab[doc[i]]
                    node2 = vocab[doc[j]]
                    if G.has_edge(node1, node2):
                        G[node1][node2]['weight'] += 1
                    else:
                        G.add_edge(node1, node2, weight=1)
        
        ##################
        graphs.append(G)
    return graphs


# Create graph-of-words representations
G_train_nx = create_graphs_of_words(train_data, vocab, 3) 
G_test_nx = create_graphs_of_words(test_data, vocab, 3)

print("Example of graph-of-words representation of document")
nx.draw_networkx(G_train_nx[3], with_labels=True)
plt.show()


from grakel.utils import graph_from_networkx
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score



# Task 12

# Transform networkx graphs to grakel representations
G_train = list(graph_from_networkx(G_train_nx, as_Graph=True, node_labels_tag='label'))
G_test = list(graph_from_networkx(G_test_nx, as_Graph=True, node_labels_tag='label'))

# Initialize a Weisfeiler-Lehman subtree kernel
gk = WeisfeilerLehman(n_iter=5, base_graph_kernel=VertexHistogram)

# Construct kernel matrices
K_train = gk.fit_transform(G_train) 
K_test = gk.transform(G_test) 

print("Kernel matrix (train):", K_train.shape)
print("Kernel matrix (test):", K_test.shape)

#Task 13

# Train an SVM classifier and make predictions

##################
# Train an SVM classifier
clf = SVC(kernel='precomputed')
clf.fit(K_train, y_train)
y_pred = clf.predict(K_test)
##################

# Evaluate the predictions
print("Accuracy:", accuracy_score(y_pred, y_test))


#Task 14


##################
from grakel.kernels import NeighborhoodSubgraphPairwiseDistance, GraphletSampling, ShortestPath

G_train = list(graph_from_networkx(G_train_nx, as_Graph=True, node_labels_tag='label'))
G_test = list(graph_from_networkx(G_test_nx, as_Graph=True, node_labels_tag='label'))

wl_kernel = WeisfeilerLehman(n_iter=5, base_graph_kernel=VertexHistogram)
shortest_path_kernel = ShortestPath(with_labels=True)
graphlet_kernel = GraphletSampling(k=3)


def train_and_evaluate(kernel, G_train, G_test, y_train, y_test):
    K_train = kernel.fit_transform(G_train)
    K_test = kernel.transform(G_test)
    clf = SVC(kernel='precomputed')
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test)
    return accuracy_score(y_test, y_pred)


# Evaluate SVM for the different Kernel
acc_graphlet = train_and_evaluate(graphlet_kernel, G_train, G_test, y_train, y_test)
print(f"Accuracy of Graphlet Kernel: {acc_graphlet:.4f}")
acc_wl = train_and_evaluate(wl_kernel, G_train, G_test, y_train, y_test)
print(f"Accuracy of Weisfeiler-Lehman Kernel: {acc_wl:.4f}")
acc_shortest_path = train_and_evaluate(shortest_path_kernel, G_train, G_test, y_train, y_test)
print(f"Accuracy of Shortest Path Kernel: {acc_shortest_path:.4f}")


##################
