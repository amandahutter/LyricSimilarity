import torch.nn as nn

class HistogramModel(nn.Module):
    def __init__(self, num_words, hidden_size):
        super(HistogramModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(num_words, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.model.forward(x)