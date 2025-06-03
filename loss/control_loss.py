import torch
import numpy as np

from torch import nn
from tool.config import Configuration

class ControlLoss(nn.Module):
    def __init__(self, cfg: Configuration):
        super(ControlLoss, self).__init__()
        self.cfg = cfg
        self.l1_loss = nn.SmoothL1Loss()

    def forward(self, pred, data):
        # pred: [B, 4*3 + 1] where 3 = [acc, steer, reverse_gear]
        gt_control = data['gt_control'].cuda()  # [B, 4*3 + 1]
        return self.l1_loss(pred, gt_control)
    
class ControlValLoss(nn.Module):
    def __init__(self, cfg: Configuration):
        super(ControlValLoss, self).__init__()
        self.cfg = cfg
        self.l1_loss = nn.SmoothL1Loss()

    def forward(self, pred, data):
        gt_control = data['gt_control'].cuda()
        #print("Shape of Ground Truth Control:", gt_control.shape)

        pred_throttle = pred[:, 0::3].reshape(-1) 
        pred_steer = pred[:, 1::3].reshape(-1)  
        pred_reverse = pred[:, 2::3].reshape(-1) 

        gt_throttle = gt_control[:, 0::3].reshape(-1)  
        gt_steer = gt_control[:, 1::3].reshape(-1) 
        gt_reverse = gt_control[:, 2::3].reshape(-1) 

        throttle_val_loss = self.l1_loss(pred_throttle, gt_throttle)  
        steer_val_loss = self.l1_loss(pred_steer, gt_steer) 
        reverse_val_loss = self.l1_loss(pred_reverse, gt_reverse)  

        return throttle_val_loss, steer_val_loss, reverse_val_loss