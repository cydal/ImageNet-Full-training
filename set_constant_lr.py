#!/usr/bin/env python3
"""
Set checkpoint to use constant LR (no scheduler).
"""
import torch
import sys

def set_constant_lr(input_ckpt, output_ckpt, constant_lr):
    """Set checkpoint to use constant LR."""
    print(f"Loading checkpoint: {input_ckpt}")
    ckpt = torch.load(input_ckpt, map_location='cpu')
    
    current_epoch = ckpt.get('epoch', 0)
    print(f"Current epoch: {current_epoch}")
    print(f"Setting constant LR: {constant_lr}")
    
    # Update optimizer state with constant LR
    if 'optimizer_states' in ckpt and len(ckpt['optimizer_states']) > 0:
        for opt_state in ckpt['optimizer_states']:
            for param_group in opt_state['param_groups']:
                old_lr = param_group['lr']
                param_group['lr'] = constant_lr
                param_group['initial_lr'] = constant_lr
                print(f"Optimizer LR: {old_lr:.8f} → {constant_lr:.8f}")
    
    # Update hyperparameters to use constant scheduler
    if 'hyper_parameters' in ckpt:
        ckpt['hyper_parameters']['lr'] = constant_lr
        ckpt['hyper_parameters']['lr_scheduler'] = 'constant'  # Use constant scheduler
        ckpt['hyper_parameters']['warmup_epochs'] = 0  # No warmup
        print(f"Updated hyperparameters:")
        print(f"  lr: {constant_lr}")
        print(f"  lr_scheduler: constant")
        print(f"  warmup_epochs: 0")
    
    # Remove scheduler state so it reinitializes as constant
    if 'lr_schedulers' in ckpt:
        print(f"Removing scheduler state (will use constant LR)")
        ckpt['lr_schedulers'] = []
    
    print(f"\nSaving checkpoint: {output_ckpt}")
    torch.save(ckpt, output_ckpt)
    print("Done!")
    print(f"\nThe model will train with constant LR = {constant_lr} from epoch {current_epoch} onwards")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python set_constant_lr.py <input_checkpoint> <output_checkpoint> <constant_lr>")
        print("Example: python set_constant_lr.py epoch249.ckpt epoch249_constant.ckpt 3e-5")
        sys.exit(1)
    
    input_ckpt = sys.argv[1]
    output_ckpt = sys.argv[2]
    constant_lr = float(sys.argv[3])
    
    set_constant_lr(input_ckpt, output_ckpt, constant_lr)
