import math

import torch
import torch.nn.functional as F

from torch import nn
from tool.config import Configuration


class Seg3dHead(nn.Module):
    def __init__(self, cfg: Configuration):
        super(Seg3dHead, self).__init__()
        self.cfg = cfg

        self.in_channel = self.cfg.conet_encoder_out_channel
        self.out_channel = self.cfg.conet_encoder_in_channel
        self.seg_classes = self.cfg.seg_classes
        self.seg_dim = self.cfg.seg_dim
        self.occ_size = self.cfg.occ_size

        self.relu = nn.ReLU(inplace=True)
        self.up_sample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.c5_conv = nn.Conv3d(self.in_channel, self.out_channel, (1, 1, 1))
        self.up_conv5 = nn.Conv3d(self.out_channel, self.out_channel, (1, 1, 1))
        self.up_conv4 = nn.Conv3d(self.out_channel, self.out_channel, (1, 1, 1))
        self.up_conv3 = nn.Conv3d(self.out_channel, self.out_channel, (1, 1, 1))
        self.up_conv2 = nn.Conv3d(self.out_channel, self.out_channel, (1, 1, 1))
        self.up_conv1 = nn.Conv3d(self.out_channel, self.out_channel, (1, 1, 1))

        self.segmentation_head = nn.Sequential(
            nn.Conv3d(self.out_channel, self.out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(self.out_channel),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.out_channel, self.seg_classes, kernel_size=1, padding=0)
        )

    def top_down(self, x):
        p7 = self.relu(self.c5_conv(x))
        p6 = self.relu(self.up_conv5(self.up_sample(p7)))
        p5 = self.relu(self.up_conv4(self.up_sample(p6)))
        p4 = self.relu(self.up_conv3(self.up_sample(p5)))
        p3 = self.relu(self.up_conv2(self.up_sample(p4)))
        p2 = self.relu(self.up_conv1(self.up_sample(p3)))
        p1 = F.interpolate(p2, size=tuple(self.occ_size), mode="trilinear", align_corners=False)
        return p1

    def forward(self, fuse_feature):
        fuse_feature_t = fuse_feature.transpose(1, 2)
        b, c, s = fuse_feature_t.shape
        fuse_bev = torch.reshape(fuse_feature_t, (b,c)+tuple(self.seg_dim))
        fuse_bev = self.top_down(fuse_bev)
        pred_segmentation = self.segmentation_head(fuse_bev)
        return pred_segmentation
