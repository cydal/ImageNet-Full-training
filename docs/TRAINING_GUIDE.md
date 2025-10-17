# ImageNet Training Guide

## Quick Start: Single GPU Full Training

### 1. Setup W&B (One-time)

```bash
./setup_wandb.sh
# Enter your W&B API key when prompted
# Get it from: https://wandb.ai/authorize
```

### 2. Verify FSx is Mounted

```bash
# Check FSx mount
ls -la /fsx/ns1/train/ | head
ls -la /fsx/ns1/val/ | head

# Should show 1000 class directories
```

### 3. Start Training

```bash
# Single GPU full training (100 epochs)
python train.py --config configs/single_gpu_full.yaml

# With custom W&B project name
python train.py \
    --config configs/single_gpu_full.yaml \
    --wandb_project imagenet-resnet50 \
    --wandb_name single-gpu-run-1
```

## Training Configurations

### Single GPU Full Training (Recommended for Pipeline Testing)
**Config:** `configs/single_gpu_full.yaml`

- **Epochs:** 100
- **Batch size:** 128 (per GPU)
- **Learning rate:** 0.25 (scaled for batch size)
- **Expected accuracy:** ~77-78% top-1
- **Training time:** ~7-10 days on V100/A100
- **Purpose:** Verify full pipeline before multi-node scaling

**Key features:**
- ResNet Strikes Back augmentation (mixup, cutmix, label smoothing)
- Mixed precision (FP16) for speed
- Cosine LR schedule with warmup
- Reduced weight decay (2e-5)

### ResNet Strikes Back (Full Recipe)
**Config:** `configs/resnet_strikes_back.yaml`

- **Epochs:** 600 (full recipe)
- **Batch size:** 256
- **Learning rate:** 0.5
- **Expected accuracy:** ~80.4% top-1
- **Training time:** ~40-60 days on single GPU
- **Purpose:** State-of-the-art accuracy (for multi-GPU)

## Configuration Details

### Learning Rate Scaling

The learning rate scales linearly with batch size:
```
lr = base_lr * (batch_size / 256)
```

Examples:
- Batch size 128: `lr = 0.5 * (128/256) = 0.25`
- Batch size 256: `lr = 0.5`
- Batch size 512: `lr = 0.5 * (512/256) = 1.0`
- Batch size 1024: `lr = 0.5 * (1024/256) = 2.0`

### Augmentation Strategy (ResNet Strikes Back)

1. **Random Resized Crop:** 224x224, scale (0.08, 1.0)
2. **Random Horizontal Flip:** 50% probability
3. **Mixup:** α = 0.2 (blend two images)
4. **CutMix:** α = 1.0 (cut and paste patches)
5. **Label Smoothing:** ε = 0.1

### Optimizer Settings

```yaml
optimizer: sgd
lr: 0.5  # Base for batch_size=256
momentum: 0.9
weight_decay: 2e-5  # Lower than standard 1e-4
lr_scheduler: cosine
warmup_epochs: 5
```

### Mixed Precision

```yaml
precision: 16-mixed  # FP16 training
```

Benefits:
- 2-3x faster training
- ~50% less GPU memory
- Minimal accuracy loss (<0.1%)

## Monitoring Training

### W&B Dashboard

After starting training, W&B will print a URL:
```
wandb: 🚀 View run at https://wandb.ai/your-username/imagenet-resnet50/runs/xxxxx
```

**Metrics tracked:**
- `train/loss` - Training loss
- `train/acc1` - Training top-1 accuracy
- `train/acc5` - Training top-5 accuracy
- `val/loss` - Validation loss
- `val/acc1` - Validation top-1 accuracy (main metric)
- `val/acc5` - Validation top-5 accuracy
- `lr` - Learning rate (per epoch)

### Local Logs

```bash
# View training logs
tail -f logs/train_*.log

# Check checkpoints
ls -lh checkpoints/
```

## Expected Results

### Single GPU (100 epochs)

| Epoch | Train Acc@1 | Val Acc@1 | Val Acc@5 |
|-------|-------------|-----------|-----------|
| 10    | ~35%        | ~30%      | ~55%      |
| 25    | ~55%        | ~50%      | ~75%      |
| 50    | ~70%        | ~65%      | ~85%      |
| 75    | ~75%        | ~72%      | ~90%      |
| 100   | ~78%        | ~77%      | ~93%      |

### ResNet Strikes Back (600 epochs)

| Epoch | Val Acc@1 | Val Acc@5 |
|-------|-----------|-----------|
| 100   | ~77%      | ~93%      |
| 300   | ~79%      | ~94%      |
| 600   | ~80.4%    | ~95%      |

## Troubleshooting

### Out of Memory (OOM)

Reduce batch size:
```bash
python train.py \
    --config configs/single_gpu_full.yaml \
    --batch_size 64  # Reduce from 128
    --lr 0.125  # Scale LR accordingly
```

### Slow Data Loading

Increase workers:
```bash
python train.py \
    --config configs/single_gpu_full.yaml \
    --num_workers 16  # Increase from 8
```

### Training Diverges (Loss = NaN)

1. Check learning rate (might be too high)
2. Disable mixed precision: `--precision 32`
3. Add gradient clipping: Edit config to set `gradient_clip_val: 1.0`

### W&B Not Logging

```bash
# Check W&B status
wandb status

# Re-login
wandb login

# Or disable W&B
python train.py --config configs/single_gpu_full.yaml --no_wandb
```

## Checkpointing

Checkpoints are saved to `checkpoints/` directory:
```
checkpoints/
├── resnet50-epoch=99-val_acc1=0.7745.ckpt  # Best checkpoint
├── last.ckpt                                # Latest checkpoint
└── ...
```

### Resume Training

```bash
python train.py \
    --config configs/single_gpu_full.yaml \
    --resume checkpoints/last.ckpt
```

## Next Steps: Multi-Node Training

Once single GPU training is working:

1. **Verify results:** Check W&B dashboard, ensure metrics look good
2. **Test checkpoint:** Verify checkpoint can be loaded and resumed
3. **Scale to multi-GPU:** Use `configs/resnet_strikes_back.yaml`
4. **Multi-node setup:** See `docs/MULTI_NODE_TRAINING.md`

## Performance Tips

### Single GPU Optimization

1. **Use mixed precision:** `precision: 16-mixed` (2-3x speedup)
2. **Optimize batch size:** Find max batch size that fits in GPU memory
3. **Persistent workers:** `persistent_workers: true` (faster data loading)
4. **Pin memory:** `pin_memory: true` (faster CPU-GPU transfer)

### Expected Training Speed

**Single V100 (32GB):**
- Batch size 128: ~2.5 hours/epoch
- Full 100 epochs: ~10 days

**Single A100 (40GB/80GB):**
- Batch size 128: ~1.5 hours/epoch
- Batch size 256: ~2 hours/epoch
- Full 100 epochs: ~6-8 days

## References

- [ResNet Strikes Back](https://arxiv.org/abs/2110.00476) - Training recipe
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) - Training framework
- [Weights & Biases](https://docs.wandb.ai/) - Experiment tracking
