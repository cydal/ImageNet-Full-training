# Continuation Training Plan: Epochs 300-600

## Current Status
- **Epoch:** ~300
- **Accuracy:** 76.1%
- **Goal:** Push to 78-79% (or closer to 80%)

## Strategy Summary

### 1. Learning Rate
- **Peak LR:** 1e-4 (50x lower than original 5e-3)
- **Warmup:** 10 epochs (1e-6 → 1e-4)
- **Schedule:** Cosine decay over 290 epochs
- **Min LR:** 1e-6 (never stops learning)
- **Why:** Conservative restart won't destabilize training

### 2. Augmentation (Aggressive)
- **Random Erasing:** 0.25 → 0.35 (max tested by RSB)
- **Mixup:** 0.2 → 0.3 (within RSB range)
- **CutMix:** 1.0 (unchanged)
- **Why:** Push generalization harder

### 3. Gradient Accumulation
- **Batches:** 2 (effective batch 2048 × 2 = 4096)
- **Why:** Smoother gradients, better convergence
- **Cost:** ~15% slower per epoch

### 4. Gradient Clipping
- **Value:** 1.0 (norm)
- **Why:** Stability during LR restart

### 5. Checkpoint Location
- **Path:** `/mnt/checkpoints`
- **Why:** New storage location as requested

## Files Modified

1. ✅ `train.py` - Added support for:
   - `accumulate_grad_batches`
   - `gradient_clip_algorithm`
   - `checkpoint_dir` (configurable path)

2. ✅ `configs/resnet_continuation_300to600.yaml` - New config with all changes

## How to Resume Training

### Step 1: Ensure /mnt/checkpoints exists
```bash
sudo mkdir -p /mnt/checkpoints
sudo chown ubuntu:ubuntu /mnt/checkpoints
```

### Step 2: Find best checkpoint
```bash
ls -lh checkpoints/ | grep "76.1"
# Or use the one you uploaded to S3
```

### Step 3: Start training
```bash
python train.py \
  --config configs/resnet_continuation_300to600.yaml \
  --resume checkpoints/resnet50-epoch=282-val/acc1=76.1720.ckpt \
  --wandb_project imagenet-resnet50 \
  --wandb_name RSB_A2_continuation_300to600
```

## What to Monitor

### First 20 Epochs (300-320)
**Healthy signs:**
- ✅ Loss stable or decreasing during warmup
- ✅ Val acc stays at 76.1% or improves
- ✅ No sudden spikes

**Warning signs:**
- ⚠️ Loss spikes >10% → Reduce peak LR to 5e-5
- ⚠️ Val acc drops → Reduce augmentation

### Expected Timeline
- **Epochs 300-310:** Warmup, acc ~76.1-76.3%
- **Epochs 310-400:** Steady gains, ~76.5-77.5%
- **Epochs 400-500:** Continued improvement, ~77.5-78.2%
- **Epochs 500-600:** Final refinement, **target 78.5-79%**

## Key Decisions Made

1. ✅ **No LR finder** - Conservative restart makes it unnecessary
2. ✅ **No label smoothing** - Doesn't combine well with mixup
3. ✅ **No EMA** - Working well without it (76.1%)
4. ✅ **No SAM/SWA** - Keeping it simple, staying close to RSB
5. ✅ **Aggressive augmentation** - Within RSB tested bounds
6. ✅ **Gradient accumulation** - Easy win for better convergence

## Expected Results

**Conservative estimate:** 78-79% @ epoch 600
**Optimistic:** 79-79.5% if everything goes well

To reach 80%+, would need:
- Ensemble (3-5 models)
- Or knowledge distillation
- Or architecture changes (ruled out)

## Backup Plan

If loss spikes in first 20 epochs:
1. Stop training
2. Reduce peak LR: `lr: 5.0e-5` (instead of 1e-4)
3. Resume from same checkpoint
4. Continue monitoring
