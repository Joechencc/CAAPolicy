import torch
import numpy as npn
from torch import nn
import torch.nn.functional as F
from tool.config import Configuration
import matplotlib.pyplot as plt
import math

class AttentionLoss(nn.Module):
    def __init__(self, cfg: Configuration):
        super(AttentionLoss, self).__init__()
        self.cfg = cfg
        self.eps = 1e-10
        # KL Divergence Loss between two images (16*16)
        
        

    def forward(self, mask, xattention):
        """
        Compute KL-Divergence between two 16x16 attention maps.
        
        Args:
            mask: Target attention mask (16, 16) - from BEV (normalized)
            xattention: Model's cross-attention map (16, 16) - from transformer
        
        Returns:
            KL-Divergence loss (scalar)
        """
        # mask.size(B,1,16,16), xattention.size(B,256)
        mask = mask.squeeze(1)
        # TODO: reshape 256 xattention to (16,16) image
        xattention = torch.reshape(xattention, (xattention.shape[0], int(math.sqrt(xattention.shape[1])), -1))
        # Add small epsilon to avoid log(0)
        mask = mask.clamp_min(self.eps)
        xattention = xattention.clamp_min(self.eps)
        # Plot heatmaps for debugging
        # xattention.shape = (B,16,16)
        # mask.shape = (B,16,16)
        # TODO: Enable the plot when resume the training on a good model to visualize the attention guidance.
        self.plot_heatmaps(mask.squeeze(1)[0,:,:], xattention.squeeze(1)[0,:,:])
        xattention = xattention.to(mask.device)
        # Compute KL-Divergence: sum(mask * log(mask/xattention))
        loss = F.kl_div(
            input=xattention.log(), 
            target=mask,
            reduction='batchmean',
            log_target=False
        )
        # import pdb; pdb.set_trace()
        return loss
    
    def plot_heatmaps(self, mask, xattention):
        """
        Plot heatmaps for the target mask and cross-attention map.
        """
        mask_np = mask.detach().cpu().numpy()
        xattention_np = xattention.detach().cpu().numpy()

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        
        # Plot target mask
        axs[0].imshow(mask_np, cmap='hot', interpolation='nearest')
        axs[0].set_title("Target Mask")
        axs[0].set_xlabel("X")
        axs[0].set_ylabel("Y")
        axs[0].colorbar = plt.colorbar(axs[0].imshow(mask_np, cmap='hot'), ax=axs[0])

        # Plot cross-attention map
        axs[1].imshow(xattention_np, cmap='hot', interpolation='nearest')
        axs[1].set_title("Cross-Attention Map")
        axs[1].set_xlabel("X")
        axs[1].set_ylabel("Y")
        axs[1].colorbar = plt.colorbar(axs[1].imshow(xattention_np, cmap='hot'), ax=axs[1])

        plt.tight_layout()
        plt.show()