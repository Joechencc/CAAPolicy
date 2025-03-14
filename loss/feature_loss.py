import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from tool.config import Configuration
from model.DINOv2_extractor import FeatureExtractor

class FeatureLoss(nn.Module):
    def __init__(self, cfg: Configuration):
        # Call the parent class's __init__ method
        super(FeatureLoss, self).__init__()
        
        self.cfg = cfg
        self.dinov2 = FeatureExtractor(
            model_name='resnet18',
            output_dim=512,
            image_size=256  # Must match actual input size
        )
        
    def forward(self, pred_img_feature, batch):
        future_img = batch['image']
        gt_img_feature = self.dinov2(future_img)
        B, C, H, W = gt_img_feature.shape
        gt_img_feature = gt_img_feature.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, H, W, C) -> (B, H*W, C)
        feature_loss = F.mse_loss(gt_img_feature, pred_img_feature).mean()
        return feature_loss