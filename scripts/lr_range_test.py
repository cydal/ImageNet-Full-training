#!/usr/bin/env python3
"""
Learning Rate Range Test
Quickly test different learning rates to find optimal range.
Based on: https://arxiv.org/abs/1506.01186 (Cyclical Learning Rates)
"""
import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.datamodule import ImageNetDataModule
from models.resnet50 import ResNet50Module

def lr_range_test(
    config_path: str,
    min_lr: float = 1e-6,
    max_lr: float = 10.0,
    num_iterations: int = 100,
    output_dir: str = "logs"
):
    """
    Run LR range test to find optimal learning rate.
    
    Args:
        config_path: Path to config file
        min_lr: Minimum LR to test
        max_lr: Maximum LR to test
        num_iterations: Number of iterations to test
        output_dir: Directory to save results
    """
    print("=" * 80)
    print("Learning Rate Range Test")
    print("=" * 80)
    print()
    
    # Load config
    print(f"Loading config: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print(f"Testing LR range: {min_lr:.2e} to {max_lr:.2e}")
    print(f"Iterations: {num_iterations}")
    print()
    
    # Create data module
    print("Creating data module...")
    datamodule = ImageNetDataModule(**config)
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    
    print(f"Dataset size: {len(datamodule.train_dataset):,} images")
    print(f"Batch size: {config['batch_size']}")
    print()
    
    # Create model
    print("Creating model...")
    model_config = {
        'num_classes': config['num_classes'],
        'pretrained': config['pretrained'],
        'optimizer': config['optimizer'],
        'lr': min_lr,  # Start with min_lr
        'momentum': config['momentum'],
        'weight_decay': config['weight_decay'],
        'lr_scheduler': 'constant',  # No scheduler for LR test
        'warmup_epochs': 0,
        'max_epochs': 1,
        'mixup_alpha': 0.0,  # No augmentation for LR test
        'cutmix_alpha': 0.0,
        'label_smoothing': 0.0,
        'compile_model': False,
    }
    
    model = ResNet50Module(**model_config)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Device: {device}")
    print()
    
    # Setup optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=min_lr,
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    
    # LR schedule: exponential increase from min_lr to max_lr
    lr_mult = (max_lr / min_lr) ** (1 / num_iterations)
    
    # Storage for results
    lrs = []
    losses = []
    
    print("Running LR range test...")
    print()
    
    model.train()
    iterator = iter(train_loader)
    
    for iteration in tqdm(range(num_iterations), desc="Testing LRs"):
        try:
            images, targets = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            images, targets = next(iterator)
        
        # Move to device
        images = images.to(device)
        targets = targets.to(device)
        
        # Get current LR
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = model.criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Record loss
        losses.append(loss.item())
        
        # Increase LR
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_mult
        
        # Stop if loss explodes
        if loss.item() > 10 * min(losses):
            print(f"\nStopping early at iteration {iteration} (loss exploded)")
            break
    
    print()
    print("=" * 80)
    print("Results")
    print("=" * 80)
    print()
    
    # Find optimal LR (steepest descent)
    losses = np.array(losses)
    lrs = np.array(lrs)
    
    # Smooth losses
    window = 5
    smoothed_losses = np.convolve(losses, np.ones(window)/window, mode='valid')
    smoothed_lrs = lrs[:len(smoothed_losses)]
    
    # Find steepest descent
    gradients = np.gradient(smoothed_losses)
    min_gradient_idx = np.argmin(gradients)
    optimal_lr = smoothed_lrs[min_gradient_idx]
    
    # Find LR where loss is minimum
    min_loss_idx = np.argmin(smoothed_losses)
    min_loss_lr = smoothed_lrs[min_loss_idx]
    
    print(f"Optimal LR (steepest descent): {optimal_lr:.6f}")
    print(f"LR at minimum loss: {min_loss_lr:.6f}")
    print()
    
    # Recommendations
    print("Recommendations:")
    print(f"  Conservative: {optimal_lr * 0.1:.6f} (10% of optimal)")
    print(f"  Moderate:     {optimal_lr * 0.5:.6f} (50% of optimal)")
    print(f"  Aggressive:   {optimal_lr:.6f} (optimal)")
    print()
    
    # Save plot
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Loss vs LR (log scale)
    plt.subplot(1, 2, 1)
    plt.plot(lrs, losses, alpha=0.3, label='Raw')
    plt.plot(smoothed_lrs, smoothed_losses, label='Smoothed')
    plt.axvline(optimal_lr, color='r', linestyle='--', label=f'Optimal: {optimal_lr:.6f}')
    plt.axvline(min_loss_lr, color='g', linestyle='--', label=f'Min loss: {min_loss_lr:.6f}')
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Loss vs Learning Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Loss gradient
    plt.subplot(1, 2, 2)
    plt.plot(smoothed_lrs, gradients)
    plt.axvline(optimal_lr, color='r', linestyle='--', label=f'Optimal: {optimal_lr:.6f}')
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss Gradient')
    plt.title('Loss Gradient (find steepest descent)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = output_dir / 'lr_range_test.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    # Save results
    results_path = output_dir / 'lr_range_test.txt'
    with open(results_path, 'w') as f:
        f.write("Learning Rate Range Test Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Config: {config_path}\n")
        f.write(f"LR range: {min_lr:.2e} to {max_lr:.2e}\n")
        f.write(f"Iterations: {len(lrs)}\n\n")
        f.write(f"Optimal LR (steepest descent): {optimal_lr:.6f}\n")
        f.write(f"LR at minimum loss: {min_loss_lr:.6f}\n\n")
        f.write("Recommendations:\n")
        f.write(f"  Conservative: {optimal_lr * 0.1:.6f}\n")
        f.write(f"  Moderate:     {optimal_lr * 0.5:.6f}\n")
        f.write(f"  Aggressive:   {optimal_lr:.6f}\n")
    
    print(f"Results saved to: {results_path}")
    print()
    print("=" * 80)
    print("LR Range Test Complete!")
    print("=" * 80)
    
    return optimal_lr, min_loss_lr


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Learning Rate Range Test")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum LR")
    parser.add_argument("--max_lr", type=float, default=10.0, help="Maximum LR")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    parser.add_argument("--output_dir", type=str, default="logs", help="Output directory")
    
    args = parser.parse_args()
    
    lr_range_test(
        config_path=args.config,
        min_lr=args.min_lr,
        max_lr=args.max_lr,
        num_iterations=args.iterations,
        output_dir=args.output_dir
    )
