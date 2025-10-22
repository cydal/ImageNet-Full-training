#!/usr/bin/env python3
"""
Modify checkpoint to continue training with a different LR schedule.
This sets the current LR and removes the scheduler state so it reinitializes.
"""
import torch
import sys

def modify_checkpoint(input_ckpt, output_ckpt, new_lr, new_cosine_t_max):
    """Modify checkpoint to use new LR and schedule."""
    print(f"Loading checkpoint: {input_ckpt}")
    ckpt = torch.load(input_ckpt, map_location='cpu')
    
    print(f"Original checkpoint keys: {list(ckpt.keys())}")
    print(f"Epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"Global step: {ckpt.get('global_step', 'N/A')}")
    
    # Check optimizer state
    if 'optimizer_states' in ckpt and len(ckpt['optimizer_states']) > 0:
        old_lr = ckpt['optimizer_states'][0]['param_groups'][0]['lr']
        print(f"Old LR in optimizer: {old_lr}")
        
        # Set new LR in optimizer state
        for opt_state in ckpt['optimizer_states']:
            for param_group in opt_state['param_groups']:
                param_group['lr'] = new_lr
                param_group['initial_lr'] = new_lr
        print(f"New LR set to: {new_lr}")
    
    # Remove scheduler state so it reinitializes with new config
    if 'lr_schedulers' in ckpt:
        print(f"Removing old scheduler state (had {len(ckpt['lr_schedulers'])} schedulers)")
        ckpt['lr_schedulers'] = []
    
    # Update hyperparameters if present
    if 'hyper_parameters' in ckpt:
        print(f"\nUpdating hyperparameters:")
        print(f"  Old lr: {ckpt['hyper_parameters'].get('lr', 'N/A')}")
        print(f"  Old cosine_t_max: {ckpt['hyper_parameters'].get('cosine_t_max', 'N/A')}")
        
        ckpt['hyper_parameters']['lr'] = new_lr
        ckpt['hyper_parameters']['cosine_t_max'] = new_cosine_t_max
        
        print(f"  New lr: {new_lr}")
        print(f"  New cosine_t_max: {new_cosine_t_max}")
    
    print(f"\nSaving modified checkpoint: {output_ckpt}")
    torch.save(ckpt, output_ckpt)
    print("Done!")
    print(f"\nSummary:")
    print(f"  - Epoch: {ckpt['epoch']}")
    print(f"  - Current LR: {new_lr}")
    print(f"  - New cosine T_max: {new_cosine_t_max}")
    print(f"  - Scheduler will reinitialize with new schedule")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python modify_checkpoint_lr.py <input_checkpoint> <output_checkpoint> <new_lr> <new_cosine_t_max>")
        print("Example: python modify_checkpoint_lr.py last.ckpt modified.ckpt 0.02 200")
        sys.exit(1)
    
    input_ckpt = sys.argv[1]
    output_ckpt = sys.argv[2]
    new_lr = float(sys.argv[3])
    new_cosine_t_max = int(sys.argv[4])
    
    modify_checkpoint(input_ckpt, output_ckpt, new_lr, new_cosine_t_max)
