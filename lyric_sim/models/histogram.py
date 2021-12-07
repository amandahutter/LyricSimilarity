from numpy import tanh
import torch.nn as nn

class HistogramModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(HistogramModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model.forward(x)