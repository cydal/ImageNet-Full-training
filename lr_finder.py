"""
Learning Rate Finder for ImageNet training.

Implements the LR range test to find optimal learning rate.
Based on: https://arxiv.org/abs/1506.01186

Usage:
    python lr_finder.py --config configs/stress_test.yaml
"""
import argparse
import yaml
from pathlib import Path
import matplotlib.pyplot as plt

import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner

from data.datamodule import ImageNetDataModule
from models.resnet50 import ResNet50Module
from utils.dist import setup_distributed_env


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Find optimal learning rate")
    parser.add_argument("--config", type=str, default="configs/stress_test.yaml",
                        help="Path to config file")
    parser.add_argument("--min_lr", type=float, default=1e-5,
                        help="Minimum learning rate to test")
    parser.add_argument("--max_lr", type=float, default=10.0,
                        help="Maximum learning rate to test")
    parser.add_argument("--num_training", type=int, default=100,
                        help="Number of training steps")
    
    args = parser.parse_args()
    
    # Setup distributed environment
    setup_distributed_env()
    
    # Load config
    config = load_config(args.config)
    
    # Set seed
    pl.seed_everything(config.get("seed", 42), workers=True)
    
    # Create data module
    datamodule = ImageNetDataModule(**config)
    datamodule.setup(stage="fit")
    
    # Update num_classes
    config["num_classes"] = datamodule.num_classes()
    print(f"Using {config['num_classes']} classes for LR finder")
    
    # Create model
    model = ResNet50Module(**config)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator=config.get("accelerator", "auto"),
        devices=config.get("devices", 1),
        precision=config.get("precision", "32"),
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True
    )
    
    # Run LR finder
    print(f"\nRunning LR finder from {args.min_lr} to {args.max_lr}...")
    print(f"This will take ~{args.num_training} steps\n")
    
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(
        model,
        datamodule=datamodule,
        min_lr=args.min_lr,
        max_lr=args.max_lr,
        num_training=args.num_training,
        mode="exponential"
    )
    
    # Plot results
    fig = lr_finder.plot(suggest=True)
    fig.savefig("lr_finder_result.png", dpi=150, bbox_inches='tight')
    print(f"\nLR finder plot saved to: lr_finder_result.png")
    
    # Get suggested LR
    suggested_lr = lr_finder.suggestion()
    print(f"\nSuggested learning rate: {suggested_lr:.6f}")
    print(f"Current config LR: {config.get('lr', 'not set')}")
    
    # Save results
    results = {
        'suggested_lr': float(suggested_lr),
        'config_lr': float(config.get('lr', 0)),
        'min_lr_tested': args.min_lr,
        'max_lr_tested': args.max_lr,
        'num_steps': args.num_training
    }
    
    import json
    with open('lr_finder_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: lr_finder_results.json")
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print(f"1. Suggested LR: {suggested_lr:.6f}")
    print(f"2. Conservative LR (0.5x): {suggested_lr*0.5:.6f}")
    print(f"3. Aggressive LR (1.5x): {suggested_lr*1.5:.6f}")
    print("\nUpdate your config file with the suggested LR and run training.")
    print("="*70)


if __name__ == "__main__":
    main()
