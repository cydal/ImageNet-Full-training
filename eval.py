"""
Evaluation script for ImageNet validation set.

Usage:
    python eval.py --checkpoint checkpoints/resnet50-epoch=89.ckpt --config configs/base.yaml
"""
import argparse
from pathlib import Path
import yaml

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from data.datamodule import ImageNetDataModule
from models.resnet50 import ResNet50Module


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
    parser = argparse.ArgumentParser(description="Evaluate ResNet50 on ImageNet validation set")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/base.yaml",
                        help="Path to config file")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Override data root path")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--devices", type=int, default=1,
                        help="Number of devices to use")
    parser.add_argument("--precision", type=str, default=None,
                        help="Override precision (e.g., 16-mixed, 32)")
    
    args = parser.parse_args()
    
    # Load config
    overrides = {k: v for k, v in vars(args).items() 
                 if v is not None and k not in ['config', 'checkpoint', 'devices']}
    config = load_config(args.config, overrides)
    
    # Create data module
    datamodule = ImageNetDataModule(**config)
    
    # Load model from checkpoint
    model = ResNet50Module.load_from_checkpoint(args.checkpoint)
    
    # Setup logger
    csv_logger = CSVLogger(save_dir="logs", name="eval_logs")
    
    # Create trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        precision=config.get("precision", "16-mixed") if args.precision is None else args.precision,
        logger=csv_logger,
        enable_checkpointing=False,
        enable_progress_bar=True
    )
    
    # Evaluate
    print(f"\nEvaluating checkpoint: {args.checkpoint}")
    results = trainer.validate(model, datamodule=datamodule)
    
    # Print results
    print("\n" + "="*50)
    print("Validation Results:")
    print("="*50)
    for key, value in results[0].items():
        print(f"{key}: {value:.4f}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
