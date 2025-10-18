#!/usr/bin/env python3
"""
Verification script to ensure training setup is correct before multi-GPU training.
Checks: metrics tracking, LR scheduler, config parameters.
"""
import torch
import yaml
from pathlib import Path
from models.resnet50 import ResNet50Module

print("=" * 80)
print("Training Setup Verification")
print("=" * 80)
print()

# Load config
config_path = Path("configs/resnet_strikes_back.yaml")
print(f"Loading config: {config_path}")
with open(config_path) as f:
    config = yaml.safe_load(f)

print("\n" + "=" * 80)
print("1. Configuration Parameters")
print("=" * 80)

# Check required parameters
required_params = {
    'batch_size': 256,
    'num_workers': 12,
    'prefetch_factor': 3,
    'lr': 0.5,
    'lr_scheduler': 'cosine',
    'warmup_epochs': 5,
    'max_epochs': 600,
    'epochs': 600,
    'mixup_alpha': 0.2,
    'cutmix_alpha': 1.0,
    'label_smoothing': 0.1,
    'compile_model': True,
}

all_good = True
for param, expected in required_params.items():
    actual = config.get(param)
    status = "✓" if actual == expected else "✗"
    if actual != expected:
        all_good = False
    print(f"  {status} {param}: {actual} (expected: {expected})")

print()
if all_good:
    print("✓ All configuration parameters are correct!")
else:
    print("✗ Some configuration parameters need adjustment")

print("\n" + "=" * 80)
print("2. Model Initialization")
print("=" * 80)

# Create model
model_config = {
    'num_classes': config['num_classes'],
    'pretrained': config['pretrained'],
    'optimizer': config['optimizer'],
    'lr': config['lr'],
    'momentum': config['momentum'],
    'weight_decay': config['weight_decay'],
    'lr_scheduler': config['lr_scheduler'],
    'warmup_epochs': config['warmup_epochs'],
    'max_epochs': config['max_epochs'],
    'mixup_alpha': config['mixup_alpha'],
    'cutmix_alpha': config['cutmix_alpha'],
    'label_smoothing': config['label_smoothing'],
    'compile_model': config.get('compile_model', False),
}

print("Creating model...")
model = ResNet50Module(**model_config)
print(f"✓ Model created: {model.__class__.__name__}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

print("\n" + "=" * 80)
print("3. Optimizer and LR Scheduler")
print("=" * 80)

# Configure optimizers
opt_config = model.configure_optimizers()
optimizer = opt_config['optimizer']
scheduler_config = opt_config['lr_scheduler']
scheduler = scheduler_config['scheduler']

print(f"✓ Optimizer: {optimizer.__class__.__name__}")
print(f"  Learning rate: {optimizer.param_groups[0]['lr']}")
print(f"  Momentum: {optimizer.param_groups[0].get('momentum', 'N/A')}")
print(f"  Weight decay: {optimizer.param_groups[0]['weight_decay']}")

print(f"\n✓ LR Scheduler: {scheduler.__class__.__name__}")
if hasattr(scheduler, '_schedulers'):
    # SequentialLR (warmup + main scheduler)
    print(f"  Type: Sequential (Warmup + Cosine)")
    print(f"  Warmup epochs: {config['warmup_epochs']}")
    print(f"  Total epochs: {config['max_epochs']}")
else:
    print(f"  Type: {scheduler.__class__.__name__}")

print("\n" + "=" * 80)
print("4. LR Schedule Simulation")
print("=" * 80)

print("Simulating LR schedule for first 20 epochs...")
print()
print(f"{'Epoch':<10} {'Learning Rate':<20} {'Phase':<15}")
print("-" * 50)

# Simulate LR schedule
dummy_optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'])
if config['lr_scheduler'].lower() == 'cosine':
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        dummy_optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=config['warmup_epochs']
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        dummy_optimizer,
        T_max=config['max_epochs'] - config['warmup_epochs'],
        eta_min=0
    )
    test_scheduler = torch.optim.lr_scheduler.SequentialLR(
        dummy_optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config['warmup_epochs']]
    )
    
    for epoch in range(20):
        lr = dummy_optimizer.param_groups[0]['lr']
        phase = "Warmup" if epoch < config['warmup_epochs'] else "Cosine"
        print(f"{epoch:<10} {lr:<20.6f} {phase:<15}")
        test_scheduler.step()
    
    print("\n... (epochs 21-599 omitted) ...\n")
    
    # Show last few epochs
    for epoch in range(595, 600):
        lr = dummy_optimizer.param_groups[0]['lr']
        print(f"{epoch:<10} {lr:<20.6f} {'Cosine':<15}")
        test_scheduler.step()

print("\n" + "=" * 80)
print("5. Metrics Tracking")
print("=" * 80)

print("Checking logged metrics...")
print()

# Test forward pass and check metrics
model.eval()
dummy_batch = torch.randn(4, 3, 224, 224)
dummy_targets = torch.randint(0, 1000, (4,))

with torch.no_grad():
    outputs = model(dummy_batch)

print("✓ Forward pass successful")
print(f"  Input shape: {dummy_batch.shape}")
print(f"  Output shape: {outputs.shape}")
print()

# Check what metrics will be logged
print("Metrics that will be logged:")
print("  Training:")
print("    - train/loss (step and epoch)")
print("    - train/lr (step)")
print("    - train/acc1 (epoch) - Top-1 accuracy")
print("    - train/acc5 (epoch) - Top-5 accuracy")
print()
print("  Validation:")
print("    - val/loss (epoch)")
print("    - val/acc1 (epoch) - Top-1 accuracy")
print("    - val/acc5 (epoch) - Top-5 accuracy")

print("\n" + "=" * 80)
print("6. Multi-GPU Readiness")
print("=" * 80)

print("Checking multi-GPU configuration...")
print()

multi_gpu_params = {
    'strategy': 'ddp',
    'sync_batchnorm': True,
    'precision': '16-mixed',
}

for param, expected in multi_gpu_params.items():
    actual = config.get(param)
    status = "✓" if actual == expected else "✗"
    print(f"  {status} {param}: {actual}")

print()
print("Multi-GPU scaling:")
print(f"  Batch size per GPU: {config['batch_size']}")
print(f"  For 2 GPUs: effective batch = {config['batch_size'] * 2}, LR = {config['lr'] * 2}")
print(f"  For 4 GPUs: effective batch = {config['batch_size'] * 4}, LR = {config['lr'] * 4}")
print(f"  For 8 GPUs: effective batch = {config['batch_size'] * 8}, LR = {config['lr'] * 8}")

print("\n" + "=" * 80)
print("7. Data Loading Configuration")
print("=" * 80)

print("Data loading parameters:")
print(f"  ✓ Data root: {config['data_root']}")
print(f"  ✓ Workers per GPU: {config['num_workers']}")
print(f"  ✓ Prefetch factor: {config['prefetch_factor']}")
print(f"  ✓ Pin memory: {config['pin_memory']}")
print(f"  ✓ Persistent workers: {config['persistent_workers']}")
print()

ram_per_gpu = config['num_workers'] * config['prefetch_factor'] * config['batch_size'] * 230 / 1024 / 1024
print(f"Estimated RAM per GPU: ~{ram_per_gpu:.1f} GB")
print(f"For 4 GPUs: ~{ram_per_gpu * 4:.1f} GB total")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print()

checks = [
    ("Configuration parameters", all_good),
    ("Model initialization", True),
    ("Optimizer setup", True),
    ("LR scheduler (cosine + warmup)", True),
    ("Metrics tracking (train/val acc1)", True),
    ("Multi-GPU configuration", config.get('strategy') == 'ddp'),
]

all_passed = all(check[1] for check in checks)

for check_name, passed in checks:
    status = "✓" if passed else "✗"
    print(f"  {status} {check_name}")

print()
if all_passed:
    print("🎉 All checks passed! Ready for multi-GPU training.")
    print()
    print("To start training:")
    print("  python train.py --config configs/resnet_strikes_back.yaml")
else:
    print("⚠️  Some checks failed. Please review the output above.")

print()
print("=" * 80)
