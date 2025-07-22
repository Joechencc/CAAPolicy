import os
import sys
import argparse
import yaml
from loguru import logger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from trainer.pl_dynamics_seq_trainer import DynamicsTrainingModule
from dataset.dataloader_seq_dynamic import ParkingDataModule  # Use ParkingDataModule directly
from tool.config import get_cfg

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def train():
    arg_parser = argparse.ArgumentParser(description='DynamicsModel Training')
    arg_parser.add_argument(
        '--config',
        default='./config/dynamics_seq_training.yaml',
        type=str,
        help='path to dynamics_seq_training.yaml (default: ./config/dynamics_seq_training.yaml)')
    args = arg_parser.parse_args()
    # Load configuration
    with open(args.config, 'r') as yaml_file:
        try:
            cfg_yaml = yaml.safe_load(yaml_file)
        except yaml.YAMLError:
            logger.exception("Failed to open config file: {}", args.config)
    cfg = get_cfg(cfg_yaml)

    # Set up logging
    logger.remove()
    logger.add(cfg.log_dir + '/dynamics_seq_training_{time}.log', enqueue=True, backtrace=True, diagnose=True)
    logger.add(sys.stderr, enqueue=True)
    logger.info("Config Yaml File: {}", args.config)

    # Set random seed
    seed_everything(42)

    # Initialize the data module
    parking_datamodule = ParkingDataModule(cfg)

    # Initialize the training module
    dynamics_model = DynamicsTrainingModule(cfg)

    # Set up TensorBoard logger
    tensor_logger = TensorBoardLogger(save_dir=cfg.log_dir, default_hp_metric=False)

    # Add EarlyStopping and ModelCheckpoint callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss",          # Metric to monitor
        min_delta=0.001,             # Minimum change to qualify as improvement
        mode="min",                  # Minimize the monitored metric
        stopping_threshold=1e-12,    # Stop only if val_loss is below 0.001
        patience=float('inf'),
        verbose=True
    )
    # ModelCheckpoint to save the best model
    model_checkpoint = ModelCheckpoint(
        dirpath=cfg.checkpoint_dir,  # Directory to save checkpoints
        filename="best-dynamics-{epoch:02d}-{val_loss:.4f}",  # Checkpoint filename
        save_top_k=2,               # Save only the top 2 checkpoints
        monitor="val_loss",         # Metric to monitor
        mode="min",                 # Minimize the monitored metric
        save_last=True              # Always save the last checkpoint
    )

    # Initialize the trainer
    trainer = Trainer(
        logger=tensor_logger,
        accelerator='gpu',
        devices=1,
        max_epochs=cfg.epochs,
        log_every_n_steps=cfg.log_every_n_steps,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        profiler='simple',
        callbacks=[early_stopping, model_checkpoint]  # Add callbacks here
    )

    # Optionally, set the path to your checkpoint
    resume_ckpt_path = cfg.resume_ckpt_path if hasattr(cfg, "resume_ckpt_path") else None
    if resume_ckpt_path and not os.path.exists(resume_ckpt_path):
        logger.warning(f"Checkpoint path {resume_ckpt_path} does not exist. Starting from scratch.")
        resume_ckpt_path = None

    # Train the model (resume from checkpoint if provided)
    trainer.fit(dynamics_model, datamodule=parking_datamodule, ckpt_path=resume_ckpt_path)


if __name__ == '__main__':
    train()