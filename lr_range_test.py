#!/usr/bin/env python3
"""
LR Range Test - Find optimal learning rate for fine-tuning.
Tests a range of LRs and plots loss vs LR to find the sweet spot.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

from models.resnet50 import ResNet50Module
from data.datamodule import ImageNetDataModule


def lr_range_test(
    checkpoint_path,
    data_root,
    start_lr=1e-6,
    end_lr=1e-3,
    num_iterations=100,
    batch_size=384,
    num_workers=32,
    output_dir="lr_range_test_results"
):
    """
    Run LR range test.
    
    Args:
        checkpoint_path: Path to checkpoint
        data_root: Path to ImageNet data
        start_lr: Starting learning rate
        end_lr: Ending learning rate
        num_iterations: Number of iterations to test
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers
        output_dir: Directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("LR RANGE TEST")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"LR range: {start_lr:.2e} to {end_lr:.2e}")
    print(f"Iterations: {num_iterations}")
    print(f"Batch size: {batch_size}")
    print("=" * 60)
    
    # Load checkpoint
    print("\nLoading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model
    print("Creating model...")
    model = ResNet50Module(**checkpoint['hyper_parameters'])
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()
    model.train()
    
    # Create dataloader (training set)
    print("Loading data...")
    datamodule = ImageNetDataModule(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=224,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=6
    )
    datamodule.setup('fit')
    train_loader = datamodule.train_dataloader()
    
    # Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=start_lr, momentum=0.9, weight_decay=1e-4)
    
    # LR schedule: exponential increase
    lr_mult = (end_lr / start_lr) ** (1 / num_iterations)
    
    # Storage
    lrs = []
    losses = []
    
    print("\nRunning LR range test...")
    print(f"LR multiplier per iteration: {lr_mult:.6f}")
    
    # Run test
    data_iter = iter(train_loader)
    for iteration in tqdm(range(num_iterations), desc="Testing LRs"):
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        
        images, targets = batch
        images = images.cuda()
        targets = targets.cuda()
        
        # Get current LR
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Record loss
        losses.append(loss.item())
        
        # Increase LR
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_mult
        
        # Stop if loss explodes
        if loss.item() > losses[0] * 4:
            print(f"\nStopping early at iteration {iteration} - loss exploded")
            break
    
    # Convert to numpy
    lrs = np.array(lrs)
    losses = np.array(losses)
    
    # Smooth losses for better visualization
    window_size = max(1, num_iterations // 20)
    losses_smooth = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
    lrs_smooth = lrs[:len(losses_smooth)]
    
    # Find optimal LR (steepest descent)
    gradients = np.gradient(losses_smooth)
    min_gradient_idx = np.argmin(gradients)
    optimal_lr = lrs_smooth[min_gradient_idx]
    
    # Find LR where loss is minimum
    min_loss_idx = np.argmin(losses_smooth)
    min_loss_lr = lrs_smooth[min_loss_idx]
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Steepest descent at LR: {optimal_lr:.2e}")
    print(f"Minimum loss at LR: {min_loss_lr:.2e}")
    print(f"Suggested LR: {optimal_lr/10:.2e} to {optimal_lr:.2e}")
    print("=" * 60)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Loss vs LR (log scale)
    ax1.plot(lrs, losses, alpha=0.3, label='Raw')
    ax1.plot(lrs_smooth, losses_smooth, linewidth=2, label='Smoothed')
    ax1.axvline(optimal_lr, color='r', linestyle='--', label=f'Steepest: {optimal_lr:.2e}')
    ax1.axvline(min_loss_lr, color='g', linestyle='--', label=f'Min loss: {min_loss_lr:.2e}')
    ax1.set_xlabel('Learning Rate')
    ax1.set_ylabel('Loss')
    ax1.set_xscale('log')
    ax1.set_title('LR Range Test')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss vs iteration
    ax2.plot(losses, alpha=0.5, label='Loss')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss vs Iteration')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / 'lr_range_test.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    
    # Save data
    results = {
        'lrs': lrs.tolist(),
        'losses': losses.tolist(),
        'optimal_lr': float(optimal_lr),
        'min_loss_lr': float(min_loss_lr),
        'suggested_lr_min': float(optimal_lr / 10),
        'suggested_lr_max': float(optimal_lr)
    }
    
    import json
    results_path = output_dir / 'lr_range_test_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")
    
    return optimal_lr, min_loss_lr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LR Range Test')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--data_root', type=str, default='/mnt/imagenet-data/imagenet', help='Path to ImageNet data')
    parser.add_argument('--start_lr', type=float, default=1e-6, help='Starting LR')
    parser.add_argument('--end_lr', type=float, default=1e-3, help='Ending LR')
    parser.add_argument('--num_iterations', type=int, default=100, help='Number of iterations')
    parser.add_argument('--batch_size', type=int, default=384, help='Batch size per GPU')
    parser.add_argument('--num_workers', type=int, default=32, help='Number of workers')
    parser.add_argument('--output_dir', type=str, default='lr_range_test_results', help='Output directory')
    
    args = parser.parse_args()
    
    lr_range_test(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        start_lr=args.start_lr,
        end_lr=args.end_lr,
        num_iterations=args.num_iterations,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_dir=args.output_dir
    )
