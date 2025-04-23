import torch

from torch import nn


class GradientApproximator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, in_channels, kernel_size=1)  # Output: [B, 1, L]
        )

    def forward(self, fuse_feature):
        """
        Input:  fuse_feature [B, C, L]
        Output: learned spatial mask [B, 1, L], same shape as attention map
        """
        attention_map = self.net(fuse_feature)  # [B, 1, L]
        return attention_map