"""
Training entrypoint for ImageNet with PyTorch Lightning.

Usage:
    python train.py --config configs/base.yaml
    python train.py --config configs/tiny.yaml --data_root /custom/path
"""
import argparse
import os
from pathlib import Path
import yaml

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger, CSVLogger

from data.datamodule import ImageNetDataModule
from models.resnet50 import ResNet50Module
from utils.callbacks import EMACallback
from utils.dist import setup_distributed_env


def load_config(config_path: str, overrides: dict = None) -> dict:
    """Load YAML config and apply overrides."""
    # Load base config
    base_path = Path(__file__).parent / "configs" / "base.yaml"
    with open(base_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Load specific config and merge
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, "r") as f:
                specific_config = yaml.safe_load(f)
                if specific_config:
                    config.update(specific_config)
    
    # Apply command-line overrides
    if overrides:
        config.update(overrides)
    
    return config


def main():
    parser = argparse.ArgumentParser(description="Train ResNet50 on ImageNet")
    parser.add_argument("--config", type=str, default="configs/base.yaml",
                        help="Path to config file")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Override data root path")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--precision", type=str, default=None,
                        help="Override precision (e.g., 16-mixed, 32)")
    parser.add_argument("--devices", type=int, default=None,
                        help="Number of devices to use")
    parser.add_argument("--num_nodes", type=int, default=None,
                        help="Number of nodes for distributed training")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--wandb_project", type=str, default="imagenet-training",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_name", type=str, default=None,
                        help="Weights & Biases run name")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    parser.add_argument("--max_classes", type=int, default=None,
                        help="Maximum number of classes to use (for subset training)")
    parser.add_argument("--max_samples_per_class", type=int, default=None,
                        help="Maximum samples per class (for subset training)")
    
    args = parser.parse_args()
    
    # Setup distributed environment variables
    setup_distributed_env()
    
    # Load config
    overrides = {k: v for k, v in vars(args).items() 
                 if v is not None and k not in ['config', 'resume', 'wandb_project', 'wandb_name', 'no_wandb']}
    config = load_config(args.config, overrides)
    
    # Set seed
    pl.seed_everything(config.get("seed", 42), workers=True)
    
    # Create data module
    datamodule = ImageNetDataModule(**config)
    
    # Create model
    model = ResNet50Module(**config)
    
    # Setup callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="resnet50-{epoch:02d}-{val/acc1:.4f}",
        monitor=config.get("monitor", "val/acc1"),
        mode=config.get("mode", "max"),
        save_top_k=config.get("save_top_k", 3),
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)
    
    # EMA callback (optional)
    if config.get("use_ema", False):
        ema_callback = EMACallback(decay=config.get("ema_decay", 0.9999))
        callbacks.append(ema_callback)
    
    # Setup logger
    loggers = []
    
    if not args.no_wandb:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_name,
            config=config,
            save_dir="logs"
        )
        loggers.append(wandb_logger)
    
    csv_logger = CSVLogger(save_dir="logs", name="csv_logs")
    loggers.append(csv_logger)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.get("epochs", 90),
        accelerator=config.get("accelerator", "auto"),
        devices=args.devices if args.devices else config.get("devices", "auto"),
        num_nodes=args.num_nodes if args.num_nodes else config.get("num_nodes", 1),
        strategy=config.get("strategy", "auto"),
        precision=args.precision if args.precision else config.get("precision", "32"),
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=config.get("log_every_n_steps", 50),
        val_check_interval=config.get("val_check_interval", 1.0),
        sync_batchnorm=config.get("sync_batchnorm", False),
        gradient_clip_val=config.get("gradient_clip_val", None),
        deterministic=config.get("deterministic", False),
        benchmark=not config.get("deterministic", False)
    )
    
    # Train
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)
    
    # Test on validation set
    trainer.validate(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
