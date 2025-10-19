# Training Troubleshooting Guide

## Problem: Model Not Learning (Multi-GPU)

**Symptoms:**
- Validation loss fluctuates wildly
- Accuracy stays near random (1% for 1000 classes)
- Loss doesn't decrease consistently

**Most Likely Cause: Learning Rate Too High**

### Your Case
- Multi-GPU training with LR = 0.4
- This is likely too aggressive, especially with:
  - Large batch size (256 * num_GPUs)
  - No proper warmup
  - Strong augmentation (mixup/cutmix)

## Solution: Systematic Debugging

### Step 1: Mount S3 Data (Without FSx)

FSx didn't provide speedups, so let's use direct S3 mounting:

```bash
cd /home/ubuntu/ImageNet-Full-training

# Install and mount S3
chmod +x scripts/mount_s3_direct.sh
./scripts/mount_s3_direct.sh
```

**This uses `mountpoint-s3` (AWS's official tool):**
- Faster than s3fs
- Built-in caching
- Read-only mount (safe)
- Data available at `/mnt/s3-imagenet`

**Alternative: Use local NVMe if available**
```bash
# If instance has NVMe storage
df -h | grep nvme

# Copy subset for fast iteration
mkdir -p /opt/dlami/nvme/imagenet-subset
# ... copy data
```

### Step 2: Run LR Range Test

Find optimal learning rate empirically:

```bash
cd /home/ubuntu/ImageNet-Full-training

# Run LR range test (takes ~5-10 minutes)
python scripts/lr_range_test.py \
    --config configs/troubleshoot_subset.yaml \
    --min_lr 0.0001 \
    --max_lr 1.0 \
    --iterations 100
```

**This will:**
1. Test LRs from 0.0001 to 1.0
2. Plot loss vs LR
3. Find optimal LR (steepest descent)
4. Save plot to `logs/lr_range_test.png`

**Expected output:**
```
Optimal LR (steepest descent): 0.085000
LR at minimum loss: 0.120000

Recommendations:
  Conservative: 0.008500 (10% of optimal)
  Moderate:     0.042500 (50% of optimal)
  Aggressive:   0.085000 (optimal)
```

### Step 3: Train on Subset with Conservative LR

Use the troubleshooting config with subset:

```bash
# Update config with LR from range test
# Edit configs/troubleshoot_subset.yaml, set lr: <value from test>

# Train on subset (100 classes, 50K images)
python train.py \
    --config configs/troubleshoot_subset.yaml \
    --wandb_project imagenet-debug \
    --wandb_name subset-lr-test
```

**Expected behavior (if LR is correct):**
```
Epoch 1:  val/acc1 ~5-10%   (random is 1%)
Epoch 5:  val/acc1 ~30-40%
Epoch 10: val/acc1 ~50-60%
Epoch 20: val/acc1 ~65-75%
```

**Each epoch should take ~2-3 minutes** (vs hours for full dataset)

### Step 4: Verify Learning

**Good signs:**
- ✅ Loss decreases steadily
- ✅ Train accuracy increases
- ✅ Val accuracy increases (may lag behind train)
- ✅ Val loss decreases initially

**Bad signs:**
- ❌ Loss stays constant or increases
- ❌ Accuracy stays near random (1% for 100 classes)
- ❌ Val loss fluctuates wildly
- ❌ NaN or Inf in losses

### Step 5: Scale Up Gradually

Once subset training works:

1. **Increase classes:** 100 → 500 → 1000
2. **Increase batch size:** 128 → 256
3. **Scale LR linearly:** lr_new = lr_base * (batch_new / batch_base)
4. **Add augmentation:** Enable mixup/cutmix gradually
5. **Add GPUs:** Scale LR with total batch size

## Learning Rate Guidelines

### Base LR (batch_size=256, 1 GPU)

| Scenario | Recommended LR | Notes |
|----------|----------------|-------|
| **No augmentation** | 0.1 - 0.2 | Standard SGD |
| **Basic augmentation** | 0.05 - 0.1 | Crop + flip |
| **Strong augmentation** | 0.01 - 0.05 | + mixup/cutmix |
| **Very large batch** | Use LR warmup | Gradual increase |

### LR Scaling Rules

**Linear scaling (standard):**
```
lr_scaled = lr_base * (batch_total / batch_base)

Example:
  Base: batch=256, lr=0.1
  4 GPUs: batch=1024, lr=0.4  ← This might be too high!
```

**Square root scaling (more conservative):**
```
lr_scaled = lr_base * sqrt(batch_total / batch_base)

Example:
  Base: batch=256, lr=0.1
  4 GPUs: batch=1024, lr=0.2  ← More stable
```

**Recommendation:** Start with square root scaling or lower

### Warmup is Critical for Large Batch

```yaml
# For large batch (>1024), use longer warmup
warmup_epochs: 5-10
lr: 0.2  # Start conservative
```

**Warmup schedule:**
```
Epoch 0-4: Linear 0.02 → 0.2 (10% → 100%)
Epoch 5+:  Cosine 0.2 → 0.0
```

## Common Issues and Fixes

### Issue 1: Loss Explodes

**Symptoms:** Loss becomes NaN or Inf

**Causes:**
- LR too high
- Gradient explosion
- Numerical instability

**Fixes:**
```yaml
# Reduce LR
lr: 0.01  # Much lower

# Add gradient clipping
gradient_clip_val: 1.0

# Use mixed precision carefully
precision: 16-mixed  # or 32 if issues persist
```

### Issue 2: Loss Plateaus Early

**Symptoms:** Loss stops decreasing after few epochs

**Causes:**
- LR too low
- Model underfitting
- Insufficient capacity

**Fixes:**
```yaml
# Increase LR slightly
lr: 0.15  # From 0.1

# Reduce regularization
weight_decay: 0.00001  # From 0.0001

# Train longer
epochs: 50  # From 20
```

### Issue 3: Val Loss Increases (Overfitting)

**Symptoms:** Train loss decreases, val loss increases

**Causes:**
- Overfitting on training data
- Too little regularization

**Fixes:**
```yaml
# Increase regularization
weight_decay: 0.0001  # From 0.00001

# Add augmentation
mixup_alpha: 0.2
cutmix_alpha: 1.0
label_smoothing: 0.1

# Add dropout (in model code)
```

### Issue 4: Slow Convergence

**Symptoms:** Model learns but very slowly

**Causes:**
- LR too low
- Batch size too small
- Poor data loading

**Fixes:**
```yaml
# Increase LR
lr: 0.2  # From 0.05

# Increase batch size
batch_size: 256  # From 128

# Optimize data loading
num_workers: 12
prefetch_factor: 3
```

## Debugging Checklist

Before scaling to multi-GPU:

- [ ] **Data loading works**
  - [ ] Images load correctly
  - [ ] Labels are correct
  - [ ] Augmentation applied properly
  
- [ ] **Model trains on subset**
  - [ ] Loss decreases
  - [ ] Accuracy increases
  - [ ] No NaN/Inf
  
- [ ] **LR is appropriate**
  - [ ] Ran LR range test
  - [ ] Used conservative value
  - [ ] Added warmup for large batch
  
- [ ] **Metrics tracked**
  - [ ] train/loss, train/acc1
  - [ ] val/loss, val/acc1
  - [ ] train/lr logged
  
- [ ] **Single GPU works**
  - [ ] Trains to reasonable accuracy
  - [ ] Stable training
  - [ ] GPU utilization good

## Recommended Workflow

### Phase 1: Debug (Subset, Single GPU)

```yaml
# configs/troubleshoot_subset.yaml
max_classes: 100
max_samples_per_class: 500
batch_size: 128
lr: 0.05  # Conservative
epochs: 20
mixup_alpha: 0.0  # Disabled
```

**Goal:** Verify model can learn (2-3 hours)

### Phase 2: Validate (Full Data, Single GPU)

```yaml
# configs/single_gpu_full.yaml
num_classes: 1000
batch_size: 256
lr: 0.1  # From LR test
epochs: 100
mixup_alpha: 0.2  # Enabled
```

**Goal:** Verify full pipeline (4-7 days)

### Phase 3: Scale (Multi-GPU)

```yaml
# configs/resnet_strikes_back.yaml
batch_size: 256  # Per GPU
lr: 0.5  # Scaled for 4 GPUs (0.1 * sqrt(4) ≈ 0.2, or 0.1 * 4 = 0.4)
warmup_epochs: 10  # Longer warmup
epochs: 600
```

**Goal:** Full training (6-11 days on 4 GPUs)

## S3 Mount Options

### Option 1: mountpoint-s3 (Recommended)

```bash
./scripts/mount_s3_direct.sh
# Data at: /mnt/s3-imagenet
```

**Pros:**
- Official AWS tool
- Fast
- Built-in caching
- Reliable

**Cons:**
- First access slow (S3 latency)
- Cache in /tmp (limited space)

### Option 2: s3fs (Alternative)

```bash
# Install
sudo apt-get install s3fs

# Mount
s3fs imagenet-sij /mnt/s3-imagenet \
    -o use_cache=/tmp/s3fs-cache \
    -o parallel_count=32 \
    -o multipart_size=52 \
    -o max_stat_cache_size=100000
```

**Pros:**
- FUSE filesystem
- Widely used

**Cons:**
- Slower than mountpoint-s3
- More configuration needed

### Option 3: Local Copy (Fastest)

```bash
# If you have local NVMe storage
aws s3 sync s3://imagenet-sij /opt/dlami/nvme/imagenet \
    --only-show-errors

# Update config
data_root: /opt/dlami/nvme/imagenet
```

**Pros:**
- Fastest (local NVMe speed)
- No S3 latency

**Cons:**
- Takes time to copy (~1-2 hours)
- Requires local storage space (~150GB)
- Data lost on instance termination

## Quick Start Commands

```bash
cd /home/ubuntu/ImageNet-Full-training

# 1. Mount S3
./scripts/mount_s3_direct.sh

# 2. Run LR range test
python scripts/lr_range_test.py \
    --config configs/troubleshoot_subset.yaml \
    --min_lr 0.001 \
    --max_lr 1.0

# 3. Update config with optimal LR
# Edit configs/troubleshoot_subset.yaml

# 4. Train on subset
python train.py \
    --config configs/troubleshoot_subset.yaml \
    --wandb_project imagenet-debug \
    --wandb_name subset-test

# 5. Monitor training
# Check W&B dashboard for metrics
```

## Expected Timeline

| Phase | Duration | Goal |
|-------|----------|------|
| **LR range test** | 5-10 min | Find optimal LR |
| **Subset training** | 40-60 min | Verify learning (20 epochs) |
| **Full single GPU** | 4-7 days | Validate pipeline (100 epochs) |
| **Multi-GPU** | 6-11 days | Production training (600 epochs) |

## Summary

**Root cause:** LR = 0.4 is too high for your setup

**Solution:**
1. ✅ Mount S3 directly (no FSx)
2. ✅ Run LR range test to find optimal LR
3. ✅ Train on subset (100 classes) to verify
4. ✅ Use conservative LR (0.05-0.1 for single GPU)
5. ✅ Scale LR carefully when adding GPUs (sqrt scaling)
6. ✅ Use longer warmup for large batches

**Next steps:**
1. Mount S3 data
2. Run LR range test
3. Train on subset with found LR
4. Verify model learns (acc increases, loss decreases)
5. Scale up gradually
