#!/usr/bin/env python3
"""
Extract model weights from checkpoint without optimizer/scheduler state.
This allows resuming with a different LR schedule.
"""
import torch
import sys

def extract_weights(input_ckpt, output_ckpt):
    """Extract only model weights from checkpoint."""
    print(f"Loading checkpoint: {input_ckpt}")
    ckpt = torch.load(input_ckpt, map_location='cpu')
    
    print(f"Original checkpoint keys: {list(ckpt.keys())}")
    print(f"Epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"Global step: {ckpt.get('global_step', 'N/A')}")
    
    # Create new checkpoint with model weights and required Lightning metadata
    # Include EMPTY optimizer/scheduler states so Lightning doesn't complain
    new_ckpt = {
        'state_dict': ckpt['state_dict'],
        'epoch': 0,  # Reset to 0 so training starts fresh
        'global_step': 0,  # Reset to 0
        'pytorch-lightning_version': ckpt.get('pytorch-lightning_version', '2.0.0'),
        'hyper_parameters': ckpt.get('hyper_parameters', {}),
        'loops': {},  # Empty loops to reset training state
        'callbacks': {},  # Empty callbacks
        'optimizer_states': [],  # Empty list - will be reinitialized
        'lr_schedulers': [],  # Empty list - will be reinitialized
    }
    
    print(f"\nSaving weights-only checkpoint: {output_ckpt}")
    print(f"New checkpoint keys: {list(new_ckpt.keys())}")
    torch.save(new_ckpt, output_ckpt)
    print("Done!")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_weights.py <input_checkpoint> <output_checkpoint>")
        sys.exit(1)
    
    extract_weights(sys.argv[1], sys.argv[2])
