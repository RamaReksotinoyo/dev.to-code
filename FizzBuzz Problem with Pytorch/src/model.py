import torch
from torch import nn, optim
from torch.utils.data import (Dataset, 
                              DataLoader,
                              TensorDataset)


class FizzBuzz(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(10, 100)
        self.relu = nn.ReLU()
        self.bactnorm1d = nn.BatchNorm1d(100)
        self.output = nn.Linear(100, 4)

    def forward(self, x):
        a = self.first(x)
        relu = self.relu(a)
        batcnorm = self.bactnorm1d(relu)
        return self.output(batcnorm)