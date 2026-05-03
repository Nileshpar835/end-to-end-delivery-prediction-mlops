import torch.nn as nn

def get_linear_model(input_dim):
    return nn.Linear(input_dim,1)

def get_nn_model(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim,16),
        nn.ReLU(),
        nn.Linear(16,8),
        nn.ReLU(),
        nn.Linear(8,1)
    )