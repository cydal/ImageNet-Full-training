# Start Training - Quick Guide

## Prerequisites

1. **FSx mounted** with ImageNet data at `/fsx/ns1/`
2. **Conda environment** activated: `conda activate imagenet`
3. **W&B API key** configured (optional but recommended)

## Step-by-Step

### 1. Run Pre-Flight Check

```bash
cd /home/ubuntu/ImageNet-Full-training
./preflight_check.sh
```

This verifies:
- ✓ FSx is mounted
- ✓ Data directories exist (train/val with 1000 classes each)
- ✓ GPU is available
- ✓ Python packages installed
- ✓ Sufficient disk space
- ✓ W&B configured

### 2. Setup W&B (First Time Only)

```bash
./setup_wandb.sh
```

Enter your API key from: https://wandb.ai/authorize

### 3. Start Training

```bash
# Single GPU full training (100 epochs)
python train.py --config configs/single_gpu_full.yaml

# With custom W&B project/run name
python train.py \
    --config configs/single_gpu_full.yaml \
    --wandb_project imagenet-resnet50 \
    --wandb_name run-1-single-gpu
```

### 4. Monitor Training

**W&B Dashboard:**
- URL will be printed when training starts
- Example: `https://wandb.ai/your-username/imagenet-resnet50/runs/xxxxx`

**Terminal:**
```bash
# Training will show progress bars with:
# - Current epoch
# - Training loss
# - Validation accuracy (val/acc1)
# - Learning rate
```

**Logs:**
```bash
# View logs in real-time
tail -f logs/train_*.log
```

## Training Configuration

**Single GPU Full Training** (`configs/single_gpu_full.yaml`):
- **Epochs:** 100
- **Batch size:** 128
- **Learning rate:** 0.25 (scaled for batch size)
- **Augmentation:** Mixup + CutMix + Label Smoothing
- **Precision:** FP16 (mixed precision)
- **Expected accuracy:** ~77-78% top-1
- **Training time:** ~7-10 days on V100/A100

## Expected Output

```
GPU available: True (cuda), used: 1
...
Epoch 0:  100%|████████| 10009/10009 [2:15:23<00:00, 1.23it/s, loss=6.91, v_num=xxxx]
Validation: 100%|████████| 391/391 [05:12<00:00, 1.25it/s]
Epoch 0, val/acc1: 0.0123, val/acc5: 0.0567, val/loss: 6.89

Epoch 1:  100%|████████| 10009/10009 [2:14:56<00:00, 1.24it/s, loss=6.45, v_num=xxxx]
...
```

## Common Commands

```bash
# Start training
python train.py --config configs/single_gpu_full.yaml

# Resume from checkpoint
python train.py \
    --config configs/single_gpu_full.yaml \
    --resume checkpoints/last.ckpt

# Reduce batch size (if OOM)
python train.py \
    --config configs/single_gpu_full.yaml \
    --batch_size 64 \
    --lr 0.125

# Disable W&B logging
python train.py \
    --config configs/single_gpu_full.yaml \
    --no_wandb

# Custom number of epochs
python train.py \
    --config configs/single_gpu_full.yaml \
    --epochs 50
```

## Monitoring Metrics

Key metrics to watch:

1. **val/acc1** - Validation top-1 accuracy (main metric)
   - Target: ~77-78% after 100 epochs
   
2. **train/loss** - Should decrease steadily
   - Epoch 1: ~6.5
   - Epoch 50: ~2.5
   - Epoch 100: ~1.5

3. **val/acc5** - Validation top-5 accuracy
   - Target: ~93% after 100 epochs

4. **Learning rate** - Should follow cosine schedule
   - Warmup: 0 → 0.25 (epochs 0-5)
   - Cosine decay: 0.25 → 0 (epochs 5-100)

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
python train.py --config configs/single_gpu_full.yaml --batch_size 64 --lr 0.125
```

### Training Too Slow
```bash
# Increase num_workers
python train.py --config configs/single_gpu_full.yaml --num_workers 16
```

### W&B Not Working
```bash
# Train without W&B
python train.py --config configs/single_gpu_full.yaml --no_wandb
```

### Resume After Crash
```bash
# Automatically resumes from last checkpoint
python train.py --config configs/single_gpu_full.yaml --resume checkpoints/last.ckpt
```

## Next Steps

Once single GPU training is working well:

1. ✓ Verify metrics in W&B dashboard
2. ✓ Check checkpoint can be loaded
3. → Scale to multi-GPU (4-8 GPUs)
4. → Scale to multi-node (multiple instances)

See `docs/TRAINING_GUIDE.md` for detailed information.

## Quick Reference

| Command | Purpose |
|---------|---------|
| `./preflight_check.sh` | Verify setup before training |
| `./setup_wandb.sh` | Configure W&B API key |
| `python train.py --config configs/single_gpu_full.yaml` | Start training |
| `tail -f logs/train_*.log` | View logs |
| `ls -lh checkpoints/` | Check saved checkpoints |
| `nvidia-smi` | Monitor GPU usage |

## Support

- **Training guide:** `docs/TRAINING_GUIDE.md`
- **FSx setup:** `docs/FSX_SETUP.md`
- **Multi-node:** `docs/MULTI_NODE_TRAINING.md` (coming soon)
