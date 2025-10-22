#!/usr/bin/env python3
"""
Properly fix checkpoint LR by modifying the scheduler state itself.
This ensures the scheduler continues from where we want, not from scratch.
"""
import torch
import sys
import math

def fix_checkpoint_lr(input_ckpt, output_ckpt, target_lr, remaining_epochs, eta_min):
    """Fix checkpoint to continue with specific LR and decay schedule."""
    print(f"Loading checkpoint: {input_ckpt}")
    ckpt = torch.load(input_ckpt, map_location='cpu')
    
    current_epoch = ckpt.get('epoch', 0)
    print(f"Current epoch: {current_epoch}")
    print(f"Target LR: {target_lr}")
    print(f"Remaining epochs: {remaining_epochs}")
    print(f"Minimum LR: {eta_min}")
    
    # Update optimizer state with target LR
    if 'optimizer_states' in ckpt and len(ckpt['optimizer_states']) > 0:
        for opt_state in ckpt['optimizer_states']:
            for param_group in opt_state['param_groups']:
                old_lr = param_group['lr']
                param_group['lr'] = target_lr
                param_group['initial_lr'] = target_lr
                print(f"Optimizer LR: {old_lr} → {target_lr}")
    
    # Update hyperparameters
    if 'hyper_parameters' in ckpt:
        ckpt['hyper_parameters']['lr'] = target_lr
        ckpt['hyper_parameters']['cosine_t_max'] = remaining_epochs
        ckpt['hyper_parameters']['eta_min'] = eta_min
        ckpt['hyper_parameters']['warmup_epochs'] = 0  # NO MORE WARMUP!
        print(f"Updated hyperparameters:")
        print(f"  lr: {target_lr}")
        print(f"  cosine_t_max: {remaining_epochs}")
        print(f"  eta_min: {eta_min}")
        print(f"  warmup_epochs: 0 (disabled)")
    
    # Fix the scheduler state
    if 'lr_schedulers' in ckpt and len(ckpt['lr_schedulers']) > 0:
        print(f"\nModifying scheduler state...")
        scheduler_state = ckpt['lr_schedulers'][0]
        
        # The scheduler state contains the internal state of SequentialLR and CosineAnnealingLR
        # We need to set it up so it's past warmup and at the right position in cosine decay
        
        # For SequentialLR wrapping [LinearLR (warmup), CosineAnnealingLR]
        # We want to be in the second scheduler (cosine) at step 0
        if '_schedulers' in scheduler_state:
            # This is a SequentialLR
            scheduler_state['_milestones'] = [0]  # Warmup already done
            scheduler_state['last_epoch'] = 0  # Start of cosine phase
            
            # Update the cosine scheduler (second one)
            if len(scheduler_state['_schedulers']) > 1:
                cosine_state = scheduler_state['_schedulers'][1]
                cosine_state['T_max'] = remaining_epochs
                cosine_state['eta_min'] = eta_min
                cosine_state['base_lrs'] = [target_lr]
                cosine_state['last_epoch'] = 0  # Start of this phase
                print(f"  Set cosine scheduler: T_max={remaining_epochs}, eta_min={eta_min}")
        else:
            # Simple CosineAnnealingLR
            scheduler_state['T_max'] = remaining_epochs
            scheduler_state['eta_min'] = eta_min
            scheduler_state['base_lrs'] = [target_lr]
            scheduler_state['last_epoch'] = 0
            print(f"  Set cosine scheduler: T_max={remaining_epochs}, eta_min={eta_min}")
    
    print(f"\nSaving fixed checkpoint: {output_ckpt}")
    torch.save(ckpt, output_ckpt)
    print("Done!")
    
    # Calculate expected LR at a few milestones
    print(f"\nExpected LR schedule:")
    for epoch_offset in [0, remaining_epochs//4, remaining_epochs//2, remaining_epochs-1]:
        progress = epoch_offset / remaining_epochs
        expected_lr = eta_min + (target_lr - eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
        actual_epoch = current_epoch + epoch_offset
        print(f"  Epoch {actual_epoch}: LR ≈ {expected_lr:.6f}")

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python fix_checkpoint_lr_properly.py <input> <output> <target_lr> <remaining_epochs> <eta_min>")
        print("Example: python fix_checkpoint_lr_properly.py last.ckpt fixed.ckpt 0.02 477 1e-6")
        sys.exit(1)
    
    fix_checkpoint_lr(sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]))
