# Critical Fixes Applied - ResNet Strikes Back Compliance

## Issues Fixed (Why Training Was Stuck at ~71%)

### 1. ✅ **Soft-Target Loss for Mixup/CutMix**
**Problem:** Using hard-target `CrossEntropyLoss` with mixup/cutmix
**Impact:** Caps accuracy at ~71-72%
**Fix:** Implemented `SoftTargetCrossEntropy` that accepts soft targets (probabilities)
- When mixup/cutmix enabled: Use soft-target loss
- When disabled: Use standard CrossEntropyLoss with label smoothing

### 2. ✅ **EMA (Exponential Moving Average)**
**Problem:** No EMA of model weights
**Impact:** Missing 1-2% accuracy improvement
**Fix:** Implemented EMA with decay=0.9999
- Updates EMA weights after each training step
- Uses EMA model for validation/evaluation
- Provides more stable and better-performing model

### 3. ✅ **Weight Decay on BN/Bias**
**Problem:** Applying weight decay to BatchNorm and bias parameters
**Impact:** Degrades performance, hurts generalization
**Fix:** Separated parameters into two groups:
- With weight decay: Conv weights, FC weights
- Without weight decay: BN parameters, all biases

### 4. ✅ **RandAugment**
**Problem:** Not using RandAugment
**Impact:** Weaker augmentation, worse generalization
**Fix:** Enabled `auto_augment: randaugment` in config

## Expected Impact

**Before fixes:** ~71% accuracy (stuck)
**After fixes:** ~79-80% accuracy (ResNet Strikes Back A2 target)

## Configuration Summary

```yaml
# Learning Rate
lr: 2.0e-4  # From LR range test at epoch 124
lr_scheduler: cosine
cosine_t_max: 476  # Decay from epoch 124 to 600
eta_min: 1.0e-5  # Safe minimum (10x higher than problematic 1e-6)

# Augmentation
auto_augment: randaugment
mixup_alpha: 0.2
cutmix_alpha: 1.0
label_smoothing: 0.1  # Not used (soft-target loss handles this)

# Optimization
optimizer: sgd
momentum: 0.9
weight_decay: 0.0001  # Only on conv/fc weights, NOT on BN/bias

# Model
EMA: decay=0.9999  # Validation uses EMA weights
```

## Training Restart

Starting from: **Epoch 124** (67.5% accuracy)
Target: **~80% by epoch 600**

This is a fresh start with all ResNet Strikes Back best practices properly implemented.
