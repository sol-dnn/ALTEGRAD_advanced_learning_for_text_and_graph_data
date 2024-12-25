"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2024
"""

import numpy as np

def create_train_dataset():
    n_train = 100000
    max_train_card = 10

    ############## Task 1
    
    ##################
    X_train = np.random.randint(low=1, high=11, size=(n_train, max_train_card))
    m_array = np.random.randint(low=1, high=max_train_card + 1, size=n_train)
    column_indices = np.arange(max_train_card)
    mask = column_indices < m_array[:, np.newaxis]
    X_train[mask] = 0
    y_train = X_train.sum(axis=1)
    ##################
    
    return X_train, y_train


def create_test_dataset():
    
    ############## Task 2
    
    ##################
    block_size = 10000
    digit_steps = np.arange(5, 101, 5)
    
    X_test = [np.random.randint(low=1, high=11, size=(block_size, size)) for size in digit_steps]
    y_test = [x.sum(axis=1) for x in X_test]
    ##################
    
    return X_test, y_test


