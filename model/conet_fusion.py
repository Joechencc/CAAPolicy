import torch

from torch import nn
from timm.models.layers import trunc_normal_
from tool.config import Configuration


class CONetFusion(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg
        tf_layer = nn.TransformerEncoderLayer(d_model=self.cfg.tf_en_conet_dim, nhead=self.cfg.tf_en_heads)
        self.tf_encoder = nn.TransformerEncoder(tf_layer, num_layers=self.cfg.tf_en_layers)

        total_length = self.cfg.tf_en_conet_length
        self.pos_embed = nn.Parameter(torch.randn(1, total_length, self.cfg.tf_en_conet_dim) * .02)
        self.pos_drop = nn.Dropout(self.cfg.tf_en_dropout)

        uint_dim = int(self.cfg.tf_en_conet_length / 4)
        self.motion_encoder = nn.Sequential(
            nn.Linear(self.cfg.tf_en_motion_length, uint_dim),
            nn.ReLU(inplace=True),
            nn.Linear(uint_dim, uint_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(uint_dim * 2, self.cfg.tf_en_conet_length),
            nn.ReLU(inplace=True),
        ).to(self.cfg.device)

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if 'pos_embed' in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, conet_feature, ego_motion):
        conet_feature = conet_feature.transpose(1, 2) #([2, 128, 512]) -> ([2, 1024, 256])
        # print(conet_feature.shape)

        motion_feature = self.motion_encoder(ego_motion).transpose(1, 2).expand(-1, -1, 2)
        fuse_feature = torch.cat([conet_feature, motion_feature], dim=2) #([2, 128, 516]) -> ([2, 1024, 258])
        # print(fuse_feature.shape)

        fuse_feature = self.pos_drop(fuse_feature + self.pos_embed) #([2, 128, 516]) -> ([2, 1024, 258])
        # print(fuse_feature.shape)
        # fuse_feature = self.pos_drop(conet_feature + self.pos_embed) # Test #([2, 256, 512]) -> ([2, 1024, 256])

        fuse_feature = fuse_feature.transpose(0, 1) #([128, 2, 516]) -> ([1024, 2, 258])
        # print(fuse_feature.shape)
        fuse_feature = self.tf_encoder(fuse_feature)
        # print(fuse_feature.shape)
        fuse_feature = fuse_feature.transpose(0, 1) #([2, 128, 516]) -> ([2, 1024, 258])
        # print(fuse_feature.shape)

        return fuse_feature
