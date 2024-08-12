import torch
import torch
from torch import nn
import torch.nn.functional as F


class SegmentationLoss3D(nn.Module):
    def __init__(self, class_weights):
        super(SegmentationLoss3D, self).__init__()
        self.ignore_index = 255
        self.class_weights = class_weights

    def forward(self, pred, target):
        if target.shape[1] != 1:
            raise ValueError('segmentation label must be index label with channel dim = 1')
        
        b, s, c, h, w, d = pred.shape
        pred_seg = pred.view(b * s, c, h, w, d)
        gt_seg = target.view(b * s, h, w, d)

        seg_loss = F.cross_entropy(pred_seg,
                                   gt_seg,
                                   reduction='none',
                                   ignore_index=self.ignore_index,
                                   weight=self.class_weights.to(gt_seg.device))

        return torch.mean(seg_loss)
