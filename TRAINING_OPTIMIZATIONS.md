# Training Optimizations Applied

## Problem
- GPU utilization: ~0%
- GPU memory usage: 6.5GB / 23GB (28%)
- Training speed: ~0.13 it/s (~7.5 seconds per batch)
- **Bottleneck:** Data loading from FSx

## Optimizations Applied

### 1. Increased Batch Size
**Before:** 128  
**After:** 256

**Reason:** A10G has 23GB memory, only using 6.5GB. Doubling batch size will:
- Better GPU utilization
- Faster training (fewer iterations per epoch)
- More stable gradients

**LR Adjustment:** Scaled from 0.25 → 0.5 (linear scaling rule)

### 2. Increased Data Loading Workers
**Before:** 8 workers  
**After:** 16 workers

**Reason:** FSx can handle high throughput, but needs more parallel workers to:
- Reduce data loading bottleneck
- Keep GPU fed with data
- Better utilize network bandwidth to FSx

### 3. Added Prefetch Factor
**Before:** Not set (default 2)  
**After:** 4

**Reason:** Pre-load more batches per worker:
- Workers pre-fetch 4 batches ahead
- Reduces GPU idle time waiting for data
- Better pipeline overlap between data loading and training

## Expected Improvements

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| **Batch size** | 128 | 256 |
| **Iterations/epoch** | 10,009 | 5,005 |
| **GPU utilization** | ~0% | 80-95% |
| **GPU memory** | 6.5GB | 15-18GB |
| **Training speed** | 0.13 it/s | 0.5-1.0 it/s |
| **Time per epoch** | ~21 hours | ~2-3 hours |
| **100 epochs** | ~87 days | **~10-12 days** |

## Configuration Changes

### `configs/single_gpu_full.yaml`

```yaml
# Data settings
batch_size: 256  # Was: 128
num_workers: 16  # Was: 8
prefetch_factor: 4  # New

# Training hyperparameters
lr: 0.5  # Was: 0.25 (scaled for new batch size)
```

### `data/datamodule.py`

Added `prefetch_factor` parameter support:
- Added to `__init__` parameters
- Applied to both train and val DataLoaders
- Only used when `num_workers > 0`

## How to Apply

### Option 1: Restart Training (Recommended)

```bash
# Stop current training
# Ctrl+C or kill the process

# Start with optimized config
cd /home/ubuntu/ImageNet-Full-training
python train.py \
    --config configs/single_gpu_full.yaml \
    --wandb_project imagenet-resnet50 \
    --wandb_name single-gpu-a10g-optimized
```

### Option 2: Resume from Checkpoint

```bash
# If you want to keep progress from current run
python train.py \
    --config configs/single_gpu_full.yaml \
    --wandb_project imagenet-resnet50 \
    --wandb_name single-gpu-a10g-optimized \
    --resume checkpoints/last.ckpt
```

**Note:** Resuming with different batch size may cause issues. Fresh start recommended.

## Monitoring

After restarting, monitor:

```bash
# GPU utilization (should be 80-95%)
watch -n 1 nvidia-smi

# Training speed (should be 0.5-1.0 it/s)
tail -f logs/train_*.log
```

## Additional Optimizations (If Needed)

### If Still Bottlenecked

1. **Increase batch size further:**
   ```yaml
   batch_size: 320  # or 384
   lr: 0.625  # or 0.75 (scale accordingly)
   ```

2. **Increase workers:**
   ```yaml
   num_workers: 24  # or 32
   ```

3. **Increase prefetch:**
   ```yaml
   prefetch_factor: 8
   ```

### If OOM (Out of Memory)

1. **Reduce batch size:**
   ```yaml
   batch_size: 192
   lr: 0.375
   ```

2. **Enable gradient checkpointing** (in model code)

## FSx Optimization Tips

### Check FSx Performance

```bash
# Check FSx throughput
aws fsx describe-file-systems \
    --file-system-ids fs-02386cb09beeabb62 \
    --query 'FileSystems[0].LustreConfiguration.PerUnitStorageThroughput'

# Monitor FSx metrics in CloudWatch
# - DataReadBytes
# - DataWriteBytes
# - DataReadOperations
```

### FSx Best Practices

1. ✓ **Persistent workers** - Enabled (reduces worker restart overhead)
2. ✓ **Pin memory** - Enabled (faster CPU→GPU transfer)
3. ✓ **Prefetch** - Enabled (pre-load batches)
4. ✓ **Multiple workers** - 16 workers (parallel FSx reads)

## Verification

After applying optimizations, you should see:

```
Epoch 0:   1%|▏| 50/5005 [00:50<1:23:00,  0.99it/s, v_num=xxx, train/loss_step=6.9]
```

**Key indicators:**
- ✓ Iterations: 5005 (was 10009)
- ✓ Speed: ~1.0 it/s (was 0.13 it/s)
- ✓ Time estimate: ~1.5 hours/epoch (was 21 hours)

## Summary

**Main bottleneck:** Data loading from FSx  
**Solution:** More workers + larger batches + prefetching  
**Expected speedup:** **~7-10x faster** (21 hours → 2-3 hours per epoch)  
**Total training time:** **~10-12 days** (was 87 days)

The optimizations focus on keeping the GPU fed with data by:
1. Loading data faster (more workers)
2. Pre-loading batches (prefetch)
3. Processing more data per forward pass (larger batch)
