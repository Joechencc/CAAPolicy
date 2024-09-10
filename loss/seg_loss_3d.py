import torch
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

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

    def plot_grid_gt(self, threeD_grid, save_path=None, vmax=None, layer=None):
        H, W, D = threeD_grid.shape
        threeD_grid[threeD_grid==14]=0
        twoD_map = np.max(threeD_grid, axis=2)# compress 3D-> 2D
        # twoD_map = threeD_grid[:,:,7]
        cmap = plt.cm.viridis # viridis color projection

        if vmax is None:
            vmax=np.max(twoD_map)*1.2
        plt.imshow(twoD_map, cmap=cmap, origin='upper', vmin=np.min(twoD_map), vmax=vmax) # plot 2D

        color_legend = plt.colorbar()
        color_legend.set_label('Color Legend') # legend

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

    def plot_grid_pred(self, threeD_grid, save_path=None, vmax=None, layer=None):
        threeD_grid = np.mean(threeD_grid,0)
        H, W, D = threeD_grid.shape
        twoD_map = np.max(threeD_grid, axis=2)# compress 3D-> 2D
        # twoD_map = threeD_grid[:,:,7]
        cmap = plt.cm.viridis # viridis color projection

        if vmax is None:
            vmax=np.max(twoD_map)*1.2
        plt.imshow(twoD_map, cmap=cmap, origin='upper', vmin=np.min(twoD_map), vmax=vmax) # plot 2D

        color_legend = plt.colorbar()
        color_legend.set_label('Color Legend') # legend

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()