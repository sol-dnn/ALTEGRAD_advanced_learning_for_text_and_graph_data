"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error
import torch

from utils import create_test_dataset
from models import DeepSets, LSTM

# Initializes device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
batch_size = 64
embedding_dim = 128
hidden_dim = 64

# Generates test data
X_test, y_test = create_test_dataset()
cards = [X_test[i].shape[1] for i in range(len(X_test))]
n_samples_per_card = X_test[0].shape[0]
n_digits = 11

# Retrieves DeepSets model
deepsets = DeepSets(n_digits, embedding_dim, hidden_dim).to(device)
print("Loading DeepSets checkpoint!")
checkpoint = torch.load('model_deepsets.pth.tar')
deepsets.load_state_dict(checkpoint['state_dict'])
deepsets.eval()

# Retrieves LSTM model
lstm = LSTM(n_digits, embedding_dim, hidden_dim).to(device)
print("Loading LSTM checkpoint!")
checkpoint = torch.load('model_lstm.pth.tar')
lstm.load_state_dict(checkpoint['state_dict'])
lstm.eval()

# Dict to store the results
results = {'deepsets': {'acc':[], 'mae':[]}, 'lstm': {'acc':[], 'mae':[]}}

for i in range(len(cards)):
    y_pred_deepsets = list()
    y_pred_lstm = list()
    for j in range(0, n_samples_per_card, batch_size):
        
        ############## Task 6
    
        ##################
        x_batch = torch.tensor(X_test[i][j:j+batch_size], dtype=int).to(device)
        y_pred_deepsets.append(deepsets(x_batch))
        y_pred_lstm.append(lstm(x_batch))
        ##################
        
    y_pred_deepsets = torch.cat(y_pred_deepsets)
    y_pred_deepsets = y_pred_deepsets.detach().cpu().numpy()
    
    acc_deepsets = accuracy_score(y_test[i], y_pred_deepsets.round().astype(int))
    mae_deepsets = mean_absolute_error(y_test[i], y_pred_deepsets)
    results['deepsets']['acc'].append(acc_deepsets)
    results['deepsets']['mae'].append(mae_deepsets)
    
    y_pred_lstm = torch.cat(y_pred_lstm)
    y_pred_lstm = y_pred_lstm.detach().cpu().numpy()
    
    acc_lstm = accuracy_score(y_test[i], y_pred_lstm.round().astype(int))
    mae_lstm = mean_absolute_error(y_test[i], y_pred_lstm)
    results['lstm']['acc'].append(acc_lstm)
    results['lstm']['mae'].append(mae_lstm)


############## Task 7
    
##################
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))
fig.suptitle('Deep Sets vs LSTM summing numbers', fontsize=20, y=0.97)

ax1.plot(cards, results['deepsets']['acc'], c="b", marker="+", label="DeepSets")
ax1.plot(cards, results['lstm']['acc'], c="r", marker="+", label="LSTM")
ax1.set_title("Accuracy")
ax1.set_xlabel('Cardinal')
ax1.set_ylabel('Accuracy')
ax1.legend()

ax2.plot(cards, results['deepsets']['mae'], c="b", marker="+", label="DeepSets")
ax2.plot(cards, results['lstm']['mae'], c="r", marker="+", label="LSTM")
ax2.set_title("Mean squared error")
ax2.set_xlabel('Cardinal')
ax2.set_ylabel('Mean Squared Error')
ax2.legend()

plt.show()
##################