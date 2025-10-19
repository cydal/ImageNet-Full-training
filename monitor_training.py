#!/usr/bin/env python3
"""
Monitor training progress by reading CSV logs.
Shows learning curve and current status.
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time
import sys

def plot_learning_curve(csv_path):
    """Plot training and validation metrics."""
    try:
        df = pd.read_csv(csv_path)
        
        # Filter out rows with NaN in key columns
        train_df = df[df['train/loss_epoch'].notna()].copy()
        val_df = df[df['val/acc1'].notna()].copy()
        
        if len(train_df) == 0:
            print("No training data yet...")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Training Loss
        if len(train_df) > 0:
            axes[0, 0].plot(train_df['epoch'], train_df['train/loss_epoch'], 'b-', linewidth=2)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Training Loss')
            axes[0, 0].set_title('Training Loss over Epochs')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Validation Loss
        if len(val_df) > 0:
            axes[0, 1].plot(val_df['epoch'], val_df['val/loss'], 'r-', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Validation Loss')
            axes[0, 1].set_title('Validation Loss over Epochs')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Validation Accuracy (Top-1)
        if len(val_df) > 0:
            axes[1, 0].plot(val_df['epoch'], val_df['val/acc1'], 'g-', linewidth=2, label='Top-1')
            if 'val/acc5' in val_df.columns:
                axes[1, 0].plot(val_df['epoch'], val_df['val/acc5'], 'c-', linewidth=2, label='Top-5')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy (%)')
            axes[1, 0].set_title('Validation Accuracy over Epochs')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Learning Rate
        if 'train/lr' in df.columns:
            lr_df = df[df['train/lr'].notna()].copy()
            if len(lr_df) > 0:
                axes[1, 1].plot(lr_df['step'], lr_df['train/lr'], 'm-', linewidth=2)
                axes[1, 1].set_xlabel('Step')
                axes[1, 1].set_ylabel('Learning Rate')
                axes[1, 1].set_title('Learning Rate Schedule')
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print current status
        print("\n" + "="*70)
        print("TRAINING PROGRESS")
        print("="*70)
        
        if len(val_df) > 0:
            latest = val_df.iloc[-1]
            print(f"Epoch: {int(latest['epoch'])}")
            print(f"Train Loss: {train_df.iloc[-1]['train/loss_epoch']:.4f}")
            print(f"Val Loss: {latest['val/loss']:.4f}")
            print(f"Val Acc (Top-1): {latest['val/acc1']:.2f}%")
            if 'val/acc5' in latest:
                print(f"Val Acc (Top-5): {latest['val/acc5']:.2f}%")
            
            # Show improvement
            if len(val_df) > 1:
                first = val_df.iloc[0]
                acc_improvement = latest['val/acc1'] - first['val/acc1']
                loss_improvement = first['val/loss'] - latest['val/loss']
                print(f"\nImprovement from Epoch 0:")
                print(f"  Accuracy: {acc_improvement:+.2f}%")
                print(f"  Loss: {loss_improvement:+.4f}")
        
        print("="*70)
        print("\nPlot saved to: training_progress.png")
        
    except Exception as e:
        print(f"Error: {e}")

def main():
    # Find the latest CSV log
    log_dir = Path("logs/csv_logs")
    if not log_dir.exists():
        print("No logs directory found. Training may not have started yet.")
        return
    
    # Find all version directories
    versions = sorted(log_dir.glob("version_*"))
    if not versions:
        print("No training logs found yet.")
        return
    
    latest_version = versions[-1]
    csv_file = latest_version / "metrics.csv"
    
    if not csv_file.exists():
        print(f"No metrics.csv found in {latest_version}")
        return
    
    print(f"Monitoring: {csv_file}")
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            plot_learning_curve(csv_file)
            time.sleep(30)  # Update every 30 seconds
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
        plot_learning_curve(csv_file)  # One final update

if __name__ == "__main__":
    main()
