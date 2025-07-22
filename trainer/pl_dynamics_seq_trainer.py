import torch
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader
from model.dynamics_seq_model import DynamicsModel
from tool.config import Configuration


class DynamicsTrainingModule(pl.LightningModule):
    def __init__(self, cfg: Configuration):
        super(DynamicsTrainingModule, self).__init__()
        self.cfg = cfg
        self.model = DynamicsModel()
        self.criterion = nn.MSELoss()
        self.seq_length = self.cfg.seq_length

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
        delta_mean, log_var, _, _ = self.model(data)

        # Compute loss
        # loss_per_sample = self.nll_loss(delta_mean, log_var, next_ego_pos[:, :2]- ego_pos[:, :2])
        # # Apply weighting based on speed
        # speed = torch.norm(batch['ego_motion'][:, :2], dim=1)
        # low_speed_mask = speed < 0.001  # e.g., threshold = 0.1
        # weights = torch.ones_like(speed)
        # weights[low_speed_mask] = 10  # e.g., higher_weight = 10
        # loss = (loss_per_sample * weights).mean()
        # loss = self.criterion(delta_mean, next_ego_pos[:, :2]- ego_pos[:, :2])
        loss = self.nll_loss(delta_mean, log_var, next_ego_pos[:, :2]- ego_pos[:, -1,:2])

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
        delta_mean, log_var, _, _= self.model(data)
        # Compute loss
        # loss = self.criterion(delta_mean, next_ego_pos[:, :2]- ego_pos[:, :2])
        loss = self.nll_loss(delta_mean, log_var, next_ego_pos[:, :2]- ego_pos[:, -1, :2])

        # Compute loss for coarse_prediction (comparison only)
        # coarse_loss = self.nll_loss(coarse_prediction, next_ego_pos[:, :2])

        # Log validation loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Log coarse prediction loss for comparison
        # self.log("coarse_loss", coarse_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=self.cfg.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.cfg.epochs)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def nll_loss(self, mean, log_var, target):
        var = torch.exp(log_var)
        nll = 0.5 * (log_var + ((target - mean) ** 2) / var)
        return nll.mean()