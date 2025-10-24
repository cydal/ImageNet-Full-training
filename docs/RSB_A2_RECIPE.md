# ResNet Strikes Back A2 - Full Implementation

## Complete RSB A2 Recipe (arXiv:2110.00476)

### Target Performance
- **300 epochs:** ~79.8% top-1 accuracy
- **600 epochs:** ~80.4% top-1 accuracy

### Configuration

#### Optimizer & Learning Rate
- **Optimizer:** LAMB (critical!)
- **Base LR:** 5e-3
- **Weight Decay:** 0.02
- **Warmup:** 5 epochs (linear from 0.01×LR to LR)
- **Schedule:** Cosine decay to eta_min=0 over 295 epochs
- **No weight decay on:** BatchNorm parameters, biases

#### Batch & Data
- **Global Batch:** 2048 (8 GPUs × 256 per GPU)
- **Workers:** 24 per GPU (192 total)
- **Precision:** FP16 mixed precision

#### Loss Function
- **With Mixup/CutMix:** BCE with soft targets
- **Without:** CrossEntropyLoss with label smoothing 0.1

#### Augmentations (The "Ingredients")
1. **RandAugment:** M≈7-9 (enabled)
2. **Random Erasing:** p=0.25, scale=(0.02, 0.33)
3. **Mixup:** α=0.2
4. **CutMix:** α=1.0
5. **Label Smoothing:** 0.1 (implicit in BCE with soft targets)
6. **RandomResizedCrop:** scale=(0.08, 1.0)
7. **RandomHorizontalFlip**

#### Model Enhancements
- **EMA:** decay=0.9999 (validation uses EMA weights)
- **torch.compile:** Enabled for speed
- **channels_last:** Memory format for Tensor Cores

### Key Differences from Previous Attempts

| Component | Previous | RSB A2 | Impact |
|-----------|----------|--------|--------|
| Optimizer | SGD | **LAMB** | Critical for 79.8% |
| LR | 2e-4 | **5e-3** | 25x higher |
| Weight Decay | 1e-4 | **0.02** | 200x higher |
| Loss | Soft CE | **BCE** | Better with mixup |
| Random Erasing | ❌ | **✅ p=0.25** | +1-2% |
| Batch Size | 384/GPU | **256/GPU** | Standard |
| Epochs | 600 | **300** | Faster to 79.8% |

### Why Previous Attempts Failed

1. **Wrong optimizer:** SGD instead of LAMB
2. **LR too low:** 2e-4 vs 5e-3 (25x difference!)
3. **WD too low:** 1e-4 vs 0.02 (200x difference!)
4. **Missing Random Erasing:** Critical augmentation
5. **Wrong loss:** Soft-target CE instead of BCE

### Expected Training Curve

- **Epoch 5:** ~25-30% (end of warmup)
- **Epoch 30:** ~55-60%
- **Epoch 90:** ~73-75%
- **Epoch 150:** ~77-78%
- **Epoch 300:** ~79.8% ✅ TARGET

### Starting Point

**Training from scratch** (not using epoch 124 checkpoint)
- Reason: Previous training used wrong optimizer/hyperparams
- Better to start fresh with correct recipe

### References

- Paper: https://arxiv.org/abs/2110.00476
- Key insight: LAMB + high LR + high WD + full augmentation suite = 79.8%
