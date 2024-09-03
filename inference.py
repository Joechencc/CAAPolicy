import torch
import pytorch_lightning as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, LearningRateMonitor, ModelSummary
from tool.config import Configuration
from loss.control_loss import ControlLoss, ControlValLoss
from loss.depth_loss import DepthLoss
from loss.seg_loss import SegmentationLoss
from loss.seg_loss_3d import SegmentationLoss3D
from model.parking_model import ParkingModel
import torch.nn.functional as F
from collections import OrderedDict

import os
import sys
import argparse
import yaml

from loguru import logger
from pytorch_lightning import Trainer, seed_everything
from trainer.pl_trainer import ParkingTrainingModule, setup_callbacks
from pytorch_lightning.loggers import TensorBoardLogger

from dataset.dataloader import ParkingDataModule
from tool.config import get_cfg

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class ParkingTrainingModule(pl.LightningModule):
    def __init__(self, cfg: Configuration, model_path):
        super(ParkingTrainingModule, self).__init__()
        self.save_hyperparameters()

        self.cfg = cfg

        self.control_loss_func = ControlLoss(self.cfg)

        self.control_val_loss_func = ControlValLoss(self.cfg)

        if self.cfg.feature_encoder == "bev":
            self.segmentation_loss_func = SegmentationLoss(
                class_weights=torch.Tensor(self.cfg.seg_vehicle_weights)
            )
            self.depth_loss_func = DepthLoss(self.cfg)
        elif self.cfg.feature_encoder == "conet":
            self.segmentation_loss_func_3D = SegmentationLoss3D(
                class_weights=torch.Tensor(self.cfg.seg_conet_vehicle_weights)
            )
            self.depth_loss_func = DepthLoss(self.cfg)

        self.parking_model = ParkingModel(self.cfg)
        self.load_model(model_path)

    
    def load_model(self, parking_pth_path):
        if parking_pth_path is not None:
            ckpt = torch.load(parking_pth_path)

            state_dict = OrderedDict([(k.replace('parking_model.', ''), v) for k, v in ckpt['state_dict'].items()])

            self.parking_model.load_state_dict(state_dict, strict=True)
            print('Load E2EParkingModel from %s', parking_pth_path)
            self.parking_model.eval()
        else:
            print("No pretrain model loadded")
 
    def validation_step(self, batch, batch_idx):
        val_loss_dict = {}

        pred_control, coarse_segmentation, pred_segmentation, pred_depth = self.parking_model(batch)
        self.plot_grid(coarse_segmentation, os.path.join("visual", "inference_pred_fine.png"))
        self.plot_grid(pred_segmentation, os.path.join("visual", "inference_pred_coarse.png"))
            
        acc_steer_val_loss, reverse_val_loss = self.control_val_loss_func(pred_control, batch)
        val_loss_dict.update({
            "acc_steer_val_loss": acc_steer_val_loss,
            "reverse_val_loss": reverse_val_loss
        })

        if self.cfg.feature_encoder == "bev":
            segmentation_val_loss = self.segmentation_loss_func(pred_segmentation.unsqueeze(1), batch['segmentation'])
        elif self.cfg.feature_encoder == "conet":
            if self.cfg.only_3d_perception == False:
                H,W,D = pred_segmentation.shape[-3:]
                coarse_segmentation = F.interpolate(coarse_segmentation, size=[H, W, D], mode='trilinear', align_corners=False).contiguous()
                segmentation_val_loss = self.segmentation_loss_func_3D(pred_segmentation.unsqueeze(1), batch['segmentation'])
                coarse_segmentation_val_loss = self.segmentation_loss_func_3D(coarse_segmentation.unsqueeze(1), batch['segmentation'])
            elif self.cfg.only_3d_perception == True:
                H,W,D = pred_segmentation.shape[-3:]
                coarse_segmentation = F.interpolate(coarse_segmentation, size=[H, W, D], mode='trilinear', align_corners=False).contiguous()
                segmentation_val_loss = self.segmentation_loss_func_3D(pred_segmentation.unsqueeze(1), batch['segmentation'])
                coarse_segmentation_val_loss = self.segmentation_loss_func_3D(coarse_segmentation.unsqueeze(1), batch['segmentation'])
                val_loss_dict.update({
                    "acc_steer_val_loss": 0,
                    "reverse_val_loss": 0,
                })
        val_loss_dict.update({
            "coarse_segmentation_val_loss": coarse_segmentation_val_loss,
            "segmentation_val_loss": segmentation_val_loss
        })
        
        depth_val_loss = self.depth_loss_func(pred_depth, batch['depth'])
        val_loss_dict.update({
            "depth_val_loss": depth_val_loss
        })

        val_loss = sum(val_loss_dict.values())
        val_loss_dict.update({
            "val_loss": val_loss
        })

        self.log_dict(val_loss_dict)
        # self.log_segmentation(pred_segmentation, batch['segmentation'], 'segmentation_val')
        # self.log_depth(pred_depth, batch['depth'], 'depth_val')

        return val_loss
    def plot_grid(self, threeD_grid, save_path=None, vmax=None, layer=None):
        # import pdb; pdb.set_trace()
        threeD_grid = torch.argmax(threeD_grid[0], dim=0).cpu().numpy()
        H, W, D = threeD_grid.shape


        threeD_grid[threeD_grid==4]=1
        threeD_grid[threeD_grid==17]=2
        twoD_map = np.sum(threeD_grid, axis=2) # compress 3D-> 2D
        # twoD_map = twoD_map[::-1,::-1]
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

def main():
    parser = argparse.ArgumentParser(description='Run inference with the ParkingModel')
    parser.add_argument('--config', default='./config/training_conet.yaml', type=str, help='Path to config file')
    parser.add_argument('--model_path', default='/scratch/sy3913/ParkWithUncertainty/ckpt/exp_2024_9_2_22_35_43/last.ckpt', help='Path to the pretrained model checkpoint')
    args = parser.parse_args()

    with open(args.config, 'r') as yaml_file:
        try:
            cfg_yaml = yaml.safe_load(yaml_file)
        except yaml.YAMLError:
            logger.exception("Open {} failed!", args.config)
    cfg = get_cfg(cfg_yaml)
    cfg.batch_size = 1
    model = ParkingTrainingModule(cfg,args.model_path)
    model = model.to("cuda:0")
    data_module = ParkingDataModule(cfg)
    data_module.dummmy_setup()

    data_loader = data_module.val_dataloader()

    for batch in data_loader:
        batch = {k: v.to('cuda:0') if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        predictions = model.validation_step(batch,0)
        breakpoint()

if __name__ == '__main__':
    main()
