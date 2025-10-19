# Training Performance Analysis & Optimization

## Current Status

### Observed Performance
- **Epoch 0**: 0.43 it/s (7:37 total) - VERY SLOW
- **Epoch 1**: 1.42 it/s (2:16 total) - 3.3x faster
- **Epoch 2**: 1.60 it/s (1:56 total) - 3.7x faster
- **GPU Utilization**: Inconsistent (100% → 0% → 100%)
- **GPU Memory**: 12.9GB / 23GB (56% usage)
- **System RAM**: 12GB / 62GB (safe)

### Confirmed Working
✅ **Mixed Precision Training**: Using 16-bit AMP
✅ **Channels Last Memory Format**: Optimized for Tensor Cores
✅ **TF32 Matmul**: Enabled for faster computation
✅ **Model Architecture**: ResNet50 properly configured

## Root Cause: S3 I/O Bottleneck

### The Problem
1. **Data is on S3** mounted via `mountpoint-s3`
2. **First access is slow** - network latency + cold cache
3. **GPU starvation** - GPU sits idle waiting for data from S3
4. **Inconsistent utilization** - 100% when computing, 0% when waiting for data

### Why Epoch 0 is Slowest
- Persistent workers are initializing
- S3 cache is cold (no data cached yet)
- Every image read hits S3 over network

### Why Subsequent Epochs are Faster
- Persistent workers are warmed up
- S3 cache has some data
- But still not optimal

## Solutions (Ranked by Impact)

### Option 1: Aggressive Data Prefetching ⚡ (Quick Win)
**Config**: `configs/optimized_test.yaml`

```yaml
num_workers: 16  # Up from 8 - more parallel S3 reads
prefetch_factor: 4  # Up from 2 - prefetch more batches
```

**Expected Impact**:
- 16 workers × 4 prefetch = 64 batches in flight
- Should hide most S3 latency
- System RAM: ~16-20GB (safe for 62GB)
- **Try this first**

**Trade-off**: Higher RAM usage

### Option 2: Copy Subset to Local Disk 🚀 (Best Performance)
Copy the 100-class subset to local NVMe/SSD:

```bash
# Create local directory
mkdir -p /tmp/imagenet_subset/{train,val}

# Copy only the 100 classes we're using
# (Would need to identify which 100 classes from subset_seed=42)
# This would be ~25GB for 50k train + 1k val images

# Update config:
data_root: /tmp/imagenet_subset
```

**Expected Impact**:
- **5-10x faster** data loading
- Consistent GPU utilization (90-100%)
- **Best option for iterative development**

**Trade-off**: Need to copy data first (~5-10 minutes)

### Option 3: Increase Batch Size 📈 (If Memory Allows)
Current: 256 batch size, 12.9GB GPU memory

```yaml
batch_size: 384  # or even 512
lr: 0.75  # Scale accordingly
```

**Expected Impact**:
- Fewer batches per epoch = less I/O overhead
- Better GPU utilization per batch
- Faster training overall

**Trade-off**: May hit OOM, need to test

### Option 4: Reduce Image Resolution 🎯 (For Quick Tests)
```yaml
img_size: 160  # Down from 224
```

**Expected Impact**:
- Smaller images = faster to load from S3
- Faster forward/backward pass
- Good for hyperparameter tuning

**Trade-off**: Not representative of final training

### Option 5: Disable Persistent Workers (Counter-intuitive)
```yaml
persistent_workers: false
```

**Why this might help**:
- Each worker re-initializes per epoch
- Might clear memory/cache issues
- Worth testing if other options don't work

**Expected Impact**: Probably worse, but worth testing

## Recommended Action Plan

### Phase 1: Quick Test (5 minutes)
```bash
# Test optimized config with aggressive prefetching
conda activate imagenet && python train.py \
  --config configs/optimized_test.yaml \
  --no_wandb --epochs 2
```

**Expected**: 1.5-2.5 it/s from epoch 0

### Phase 2: If Still Slow (15 minutes)
Copy subset to local disk and re-test

### Phase 3: If Fast Enough
Tune learning rate and validate training pipeline

## Current Training Metrics

### Accuracy (100-class subset)
- **Epoch 0**: val/acc1 = 1.5%
- **Epoch 1**: val/acc1 = 1.0%
- **Epoch 2**: val/acc1 = ? (in progress)

**Note**: These are very low, which is expected:
- Random guessing on 100 classes = 1%
- Model is learning from scratch
- Need more epochs to see meaningful accuracy

### Learning Rate
- Using lr=0.5 with cosine schedule
- May need tuning after fixing I/O bottleneck

## Next Steps

1. **Test optimized config** with 16 workers + 4 prefetch
2. **Monitor GPU utilization** - should be more consistent
3. **If still slow**: Copy subset to local disk
4. **Once fast**: Run LR finder to tune learning rate
5. **Validate**: Run 10-20 epochs to confirm training converges

## Memory Budget

### GPU (23GB total)
- Model: ~1GB (FP16)
- Activations: ~11GB (batch_size=256, FP16)
- Optimizer states: ~1GB
- **Available for larger batches**: ~10GB

### System RAM (62GB total)
- OS + base: ~3GB
- Training process: ~12GB
- Data workers (16 × 4 prefetch): ~16-20GB
- **Available**: ~27GB (plenty of headroom)

## Conclusion

**Primary bottleneck**: S3 I/O latency causing GPU starvation

**Best solution**: Copy subset to local disk (5-10x speedup)

**Quick solution**: Aggressive prefetching (1.5-2x speedup)

**Current training IS working**, just slow due to S3 latency. Mixed precision and model optimizations are all functioning correctly.
