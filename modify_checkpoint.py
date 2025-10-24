"""
Modify a checkpoint to prepare it for continuation training with new hyperparameters.
Key insight: Keep epoch number but remove optimizer/scheduler states so they reinitialize.
"""
import torch

def modify_checkpoint_for_continuation(input_path, output_path, new_hparams):
    """Modify checkpoint for continuation training."""
    print(f"\n{'='*70}")
    print("MODIFYING CHECKPOINT FOR CONTINUATION TRAINING")
    print(f"{'='*70}\n")
    
    # Load checkpoint
    print(f"Loading checkpoint: {input_path}")
    checkpoint = torch.load(input_path, map_location='cpu')
    
    # Show what we're starting with
    print(f"\nOriginal checkpoint:")
    print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  - Global step: {checkpoint.get('global_step', 'N/A')}")
    
    if 'hyper_parameters' in checkpoint:
        old_hp = checkpoint['hyper_parameters']
        print(f"  - Old LR: {old_hp.get('lr')}")
        print(f"  - Old warmup_epochs: {old_hp.get('warmup_epochs')}")
        print(f"  - Old max_epochs: {old_hp.get('max_epochs')}")
    
    # 1. Remove EMA model weights
    state_dict = checkpoint.get('state_dict', {})
    ema_keys = [k for k in state_dict.keys() if k.startswith('ema_model.')]
    if ema_keys:
        print(f"\n✓ Removing {len(ema_keys)} EMA model keys")
        for key in ema_keys:
            del state_dict[key]
        checkpoint['state_dict'] = state_dict
    
    # 2. Update hyperparameters with new config
    print(f"\n✓ Updating hyperparameters:")
    for key, value in new_hparams.items():
        if 'hyper_parameters' not in checkpoint:
            checkpoint['hyper_parameters'] = {}
        old_value = checkpoint['hyper_parameters'].get(key, 'N/A')
        checkpoint['hyper_parameters'][key] = value
        print(f"  - {key}: {old_value} → {value}")
    
    # 3. KEEP epoch and global_step (don't reset to 0)
    # This tells Lightning we're resuming, but without optimizer/scheduler state
    print(f"\n✓ Keeping training counters:")
    print(f"  - epoch: {checkpoint.get('epoch')} (unchanged)")
    print(f"  - global_step: {checkpoint.get('global_step')} (unchanged)")
    
    # 4. Remove LR scheduler state - this forces it to reinitialize
    if 'lr_schedulers' in checkpoint:
        print(f"\n✓ Removing lr_schedulers state (will reinitialize with new schedule)")
        del checkpoint['lr_schedulers']
    
    # 5. Remove optimizer state - this forces it to reinitialize
    if 'optimizer_states' in checkpoint:
        print(f"✓ Removing optimizer_states (will reinitialize with new LR)")
        del checkpoint['optimizer_states']
    
    # 6. Keep loops and callbacks - Lightning needs these for proper resumption
    print(f"✓ Keeping loops and callbacks state (for proper resumption)")
    
    # Save modified checkpoint
    print(f"\n✓ Saving modified checkpoint: {output_path}")
    torch.save(checkpoint, output_path)
    
    print(f"\n{'='*70}")
    print("CHECKPOINT MODIFICATION COMPLETE")
    print(f"{'='*70}\n")
    print("Modified checkpoint will:")
    print("  - Load model weights from epoch 286")
    print("  - Use NEW hyperparameters (LR 1e-4, warmup 10, etc.)")
    print("  - Reinitialize optimizer and LR scheduler from scratch")
    print("  - Continue epoch counter from 287 (for logging)")
    print()

if __name__ == "__main__":
    # New hyperparameters for continuation training
    new_hparams = {
        'lr': 1.0e-4,
        'warmup_epochs': 10,
        'max_epochs': 600,
        'cosine_t_max': 290,
        'eta_min': 1.0e-6,
        'random_erasing': 0.35,
        'mixup_alpha': 0.3,
    }
    
    input_checkpoint = "checkpoints/resnet50-epoch=286-val/acc1=76.2100.ckpt"
    output_checkpoint = "checkpoints/resnet50-epoch=286-MODIFIED-for-continuation.ckpt"
    
    modify_checkpoint_for_continuation(input_checkpoint, output_checkpoint, new_hparams)
    
    print("Resume training with:")
    print(f"  python train.py --config configs/resnet_continuation_300to600.yaml \\")
    print(f"    --resume {output_checkpoint}")
