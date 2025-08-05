import torch
import random
import numpy as np
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from dataset.carla_dataset import CarlaDataset
from tool.config import Configuration

from functools import partial


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def diffusion_collate_fn(batch, seq_len):
    """
    Batch: list of dicts from Dataset __getitem__
    """
    batch_dict = {}

    # Keys you want to resize
    traj_keys = [
        'gt_target_point_traj',
        'gt_control_traj',
        'gt_acc_traj',
        'gt_steer_traj',
        'gt_reverse_traj'
    ]

    for key in traj_keys:
        resized = [resize_trajectory(item[key], seq_len) for item in batch]  # List of [fixed_steps, D]
        batch_dict[key] = torch.stack(resized, dim=0)  # → [B, fixed_steps, D]

    # Copy over all other keys directly (not resized)
    for key in batch[0]:
        if key not in traj_keys:
            batch_dict[key] = torch.utils.data.default_collate([item[key] for item in batch])

    return batch_dict

def resize_trajectory(tensor, fixed_steps):
    """
    Resize a (T, D) tensor to (fixed_steps, D) with:
    1. Keep first and last step
    2. Uniformly sample if too long
    3. Pad with last if too short
    """
    T, D = tensor.shape
    if T == fixed_steps:
        return tensor
    elif T > fixed_steps:
        # Uniformly sample indices, keeping first and last
        idx = np.linspace(0, T - 1, fixed_steps, dtype=int)
        return tensor[idx]
    else:
        # Pad with last value
        padding = tensor[-1:].repeat(fixed_steps - T, 1)  # [fixed_steps - T, D]
        return torch.cat([tensor, padding], dim=0)

class ParkingDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg
        self.data_dir = self.cfg.data_dir
        self.train_loader = None
        self.val_loader = None

    def setup(self, stage: str):
        data_root = self.data_dir
        train_set = CarlaDataset(data_root, 1, self.cfg)
        val_set = CarlaDataset(data_root, 0, self.cfg)
        diffusion_collate = partial(diffusion_collate_fn, seq_len=self.cfg.horizon)
        self.train_loader = DataLoader(dataset=train_set,
                                       batch_size=self.cfg.batch_size,
                                       shuffle=True,
                                       num_workers=16,
                                       pin_memory=True,
                                       worker_init_fn=seed_worker,
                                       drop_last=True,
                                       collate_fn=diffusion_collate)
        self.val_loader = DataLoader(dataset=val_set,
                                     batch_size=self.cfg.batch_size,
                                     shuffle=False,
                                     num_workers=16,
                                     pin_memory=True,
                                     worker_init_fn=seed_worker,
                                     drop_last=True,
                                     collate_fn=diffusion_collate)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
