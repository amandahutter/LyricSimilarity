from numpy import tanh
import torch.nn as nn

class HistogramModel(nn.Module):
    def __init__(self, input_size, hidden_size_0, hidden_size_1):
        super(HistogramModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size_0),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_size_0, hidden_size_0),
            nn.ReLU(),
            nn.Linear(hidden_size_0, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model.forward(x)