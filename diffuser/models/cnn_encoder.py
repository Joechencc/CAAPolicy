import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    def __init__(self, in_channels=3, d_model=256, height=200, width=200, output_dim=32):
        super().__init__()
        self.d_model = d_model
        self.height = height
        self.width = width

        # Project from segmentation logits (C channels) to d_model feature size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),  # [B, 64, 100, 100]
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2), # [B, 128, 50, 50]
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=5, stride=2, padding=2),  # [B, d_model, 25, 25]
        )

        # project to a feature vector
        self.proj = nn.Sequential(
            nn.Linear(in_features=3*25*25, out_features=256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)  # For example, output size 10
        )

    def forward(self, seg_logits):
        """
        seg_logits: [B, C, H, W]  - raw segmentation logits (not normalized)
        Returns:
            memory: [B, L, d_model] - flattened encoder memory for decoder
        """
        B, C, H, W = seg_logits.shape
        assert H == self.height and W == self.width, "Unexpected input size"

        # Normalize segmentation logits → probabilities
        seg_probs = F.softmax(seg_logits, dim=1)           # [B, C, H, W]

        # Project to d_model channels
        feat = self.conv(seg_probs)                        # [B, d_model, H, W]

        # Project to a feature vector
        feat = self.proj(feat.view(B, -1))

        return feat