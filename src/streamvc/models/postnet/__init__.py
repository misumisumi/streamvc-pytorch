import torch.nn as nn


class SoftLabelPostNet(nn.Module):
    def __init__(self, channels, n_class):
        super(SoftLabelPostNet, self).__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(channels), nn.Linear(channels, n_class)
        )

    def forward(self, x):  # (B, C, T)→(B, T, C)
        out = self.layers(x.transpose(1, 2)).transpose(1, 2)
        return out
