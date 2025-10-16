# Quick Reference

## Full Training (Production)

```bash
# Standard full ImageNet training
python train.py --config configs/full.yaml

# With custom settings
python train.py \
    --config configs/full.yaml \
    --batch_size 256 \
    --epochs 90 \
    --lr 0.1
```

**Guarantee:** Uses original `ImageFolder` dataset with zero overhead from subsetting code.

## Subset Training (Testing/Development)

```bash
# Quick test with 10 classes, 50 samples each
python train.py \
    --config configs/tiny_gpu.yaml \
    --epochs 1 \
    --no_wandb \
    --max_classes 10 \
    --max_samples_per_class 50

# Larger test with 100 classes
python train.py \
    --config configs/tiny_gpu.yaml \
    --epochs 5 \
    --max_classes 100 \
    --max_samples_per_class 500
```

**Note:** Labels are automatically remapped to [0, N-1] range.

## Verification Tests

```bash
# Verify full training is unaffected
python /home/ubuntu/test_full_training_unaffected.py

# Verify subset training label remapping works
python /home/ubuntu/test_label_remapping.py
```

## Key Parameters

| Parameter | Purpose | Default | Full Training | Subset Training |
|-----------|---------|---------|---------------|-----------------|
| `--max_classes` | Limit number of classes | `None` | Don't set | Set to N (e.g., 10) |
| `--max_samples_per_class` | Limit samples per class | `None` | Don't set | Set to M (e.g., 50) |
| `--epochs` | Training epochs | From config | 90 | 1-5 |
| `--batch_size` | Batch size | From config | 256+ | 32-64 |
| `--no_wandb` | Disable W&B logging | `False` | Don't set | Set for testing |

## Performance Impact

| Mode | Dataset Type | Overhead | Use Case |
|------|--------------|----------|----------|
| **Full training** | `ImageFolder` | **0%** | Production |
| Subset training | `RemappedSubset` | ~0.1% | Testing/debugging |

## Design Guarantee

**The subsetting code will never affect full training performance.**

- Early return when `max_classes=None` and `max_samples_per_class=None`
- Original dataset returned unchanged
- Zero computational overhead
- Zero memory overhead

See [FULL_TRAINING_GUARANTEE.md](docs/FULL_TRAINING_GUARANTEE.md) for details.
