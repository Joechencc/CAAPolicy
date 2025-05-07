import torch
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader
from dataset.carla_dataset import CarlaDataset
# from model.dynamics_model import DynamicsModel
from model.hybrid_dynamics_model import HybridDynamicsModel as DynamicsModel 
from tool.config import Configuration


class DynamicsTrainingModule(pl.LightningModule):
    def __init__(self, cfg: Configuration):
        super(DynamicsTrainingModule, self).__init__()
        self.cfg = cfg
        self.model = DynamicsModel()
        self.criterion = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        # Extract inputs and targets from the batch
        ego_motion = batch['ego_motion']
        raw_control = batch['raw_control']
        ego_pos = batch['ego_pos']
        next_ego_pos = batch['ego_pos_next']

        # Forward pass
        data = {
            'ego_motion': ego_motion,
            'raw_control': raw_control,
            'ego_pos': ego_pos
        }
        pred_next_ego_pos, _, _ = self.model(data)

        # Compute loss
        loss = self.criterion(pred_next_ego_pos, next_ego_pos[:, :2])

        # Log training loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Extract inputs and targets from the batch
        ego_motion = batch['ego_motion']
        raw_control = batch['raw_control']
        ego_pos = batch['ego_pos']
        next_ego_pos = batch['ego_pos_next']

        # Forward pass
        data = {
            'ego_motion': ego_motion,
            'raw_control': raw_control,
            'ego_pos': ego_pos
        }
        pred_next_ego_pos, coarse_prediction, _ = self.model(data)
        # Compute loss
        loss = self.criterion(pred_next_ego_pos, next_ego_pos[:, :2])

        # Compute loss for coarse_prediction (comparison only)
        coarse_loss = self.criterion(coarse_prediction, next_ego_pos[:, :2])

        # Log validation loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Log coarse prediction loss for comparison
        self.log("coarse_loss", coarse_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=self.cfg.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.cfg.epochs)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}