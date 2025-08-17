import torch
import torch.nn as nn

class metricsMLP(nn.Module):
    def __init__(self, input_dim=18, hidden1=256, hidden2=128, num_classes=10):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_classes)  # logits
        )

    def forward(self, x):
        # flatten except batch dimension
        x = x.view(x.size(0), -1)
        return self.layers(x)