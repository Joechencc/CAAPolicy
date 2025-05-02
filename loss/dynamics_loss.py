import torch
import numpy as np

from torch import nn
from tool.config import Configuration


class DynamicsLoss(nn.Module):
    def __init__(self, cfg: Configuration):
        super(DynamicsLoss, self).__init__()
        self.cfg = cfg
        self.loss_fn = nn.MSELoss()  # Use Mean Squared Error as the regression loss

    def forward(self, next_ego_pos, gt_ego_pos):
        dynamics_loss = self.loss_fn(next_ego_pos, gt_ego_pos)
        return dynamics_loss
