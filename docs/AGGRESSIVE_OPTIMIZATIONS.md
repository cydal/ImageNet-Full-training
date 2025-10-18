# Aggressive Training Optimizations

## Problem Analysis

**Observed:**
- GPU utilization: 0% most of the time
- GPU memory: 13GB / 23GB (56%)
- Speed: 0.19 it/s (still slow)
- **Root cause:** Data loading bottleneck + suboptimal GPU utilization

## Aggressive Optimizations Applied

### 1. Maximized Batch Size
**Before:** 256  
**After:** 384

**Impact:**
- Uses more GPU memory (~18-20GB expected)
- Fewer iterations per epoch (5004 → 3336)
- Better GPU utilization

### 2. Maximized Data Loading
**Before:** 16 workers, prefetch_factor=4  
**After:** 32 workers, prefetch_factor=8

**Impact:**
- Maximum parallelism for FSx
- Aggressive pre-loading (256 batches buffered!)
- Should eliminate data loading bottleneck

### 3. Disabled Mixup/CutMix (Temporarily)
**Before:** mixup_alpha=0.2, cutmix_alpha=1.0  
**After:** mixup_alpha=0.0, cutmix_alpha=0.0

**Impact:**
- Removes CPU-side augmentation overhead
- **For testing only** - re-enable after confirming GPU utilization
- May reduce final accuracy by 1-2%

### 4. Added torch.compile()
**New:** `compile_model: true`

**Impact:**
- PyTorch 2.0+ graph compilation
- Fuses operations for better performance
- 10-30% speedup expected
- First epoch will be slower (compilation time)

### 5. Channels-Last Memory Format
**New:** Model and inputs use `torch.channels_last`

**Impact:**
- Better Tensor Core utilization on A10G
- Optimized for NHWC layout (vs NCHW)
- 10-20% speedup for conv-heavy models

### 6. TF32 Precision
**Already added:** `torch.set_float32_matmul_precision('medium')`

**Impact:**
- Uses TensorFloat-32 on Tensor Cores
- Faster matmul operations
- Minimal accuracy impact

## Expected Performance

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| **Batch size** | 256 | 384 |
| **Iterations/epoch** | 5,004 | 3,336 |
| **GPU utilization** | 0-10% | **70-95%** |
| **GPU memory** | 13GB | 18-20GB |
| **Training speed** | 0.19 it/s | **1.5-3.0 it/s** |
| **Time per epoch** | ~7 hours | **30-60 minutes** |
| **100 epochs** | ~29 days | **2-4 days** |

## Configuration Summary

### `configs/single_gpu_full.yaml`

```yaml
# Data loading - MAXIMIZED
batch_size: 384
num_workers: 32
prefetch_factor: 8

# Model - OPTIMIZED
compile_model: true  # PyTorch 2.0+ compilation

# Augmentation - SIMPLIFIED (for testing)
mixup_alpha: 0.0  # Disabled temporarily
cutmix_alpha: 0.0  # Disabled temporarily

# Training
lr: 0.75  # Scaled for batch_size=384
```

### `models/resnet50.py`

```python
# TF32 for Tensor Cores
torch.set_float32_matmul_precision('medium')

# Channels-last memory format
self.model = self.model.to(memory_format=torch.channels_last)
images = images.to(memory_format=torch.channels_last)

# torch.compile for graph optimization
if compile_model:
    self.model = torch.compile(self.model, mode='max-autotune')
```

## How to Test

### Start Training

```bash
cd /home/ubuntu/ImageNet-Full-training

python train.py \
    --config configs/single_gpu_full.yaml \
    --wandb_project imagenet-resnet50 \
    --wandb_name single-gpu-aggressive-v1
```

**Note:** First epoch will be slower due to torch.compile() compilation.

### Monitor Performance

```bash
# Watch GPU utilization (should be 70-95%)
watch -n 1 nvidia-smi

# Expected output after warmup:
# GPU: 85-95%
# Memory: 18-20GB / 23GB
```

### Expected Training Output

```
Epoch 0:   1%|▏| 33/3336 [00:30<50:00,  1.10it/s, v_num=xxx, train/loss_step=6.9]
```

**Key indicators:**
- ✓ Speed: 1.0-3.0 it/s (was 0.19 it/s)
- ✓ Iterations: 3336 (was 5004)
- ✓ Time estimate: ~50 min/epoch (was 7 hours)

## Troubleshooting

### If OOM (Out of Memory)

```yaml
# Reduce batch size
batch_size: 320  # or 256
lr: 0.625  # or 0.5
```

### If Still Low GPU Utilization

1. **Check data loading:**
   ```bash
   # Should see high CPU usage from workers
   htop
   ```

2. **Check FSx throughput:**
   ```bash
   # Monitor FSx metrics in CloudWatch
   # Look for throttling or limits
   ```

3. **Increase workers further:**
   ```yaml
   num_workers: 48  # if CPU allows
   ```

### If torch.compile() Fails

```yaml
# Disable compilation
compile_model: false
```

## Re-enabling Mixup/CutMix

Once GPU utilization is confirmed high (>80%), re-enable augmentation:

```yaml
mixup_alpha: 0.2
cutmix_alpha: 1.0
```

This will:
- Improve final accuracy by 1-2%
- Slightly reduce training speed
- Add CPU overhead for augmentation

## Performance Breakdown

### Expected Speedup Sources

| Optimization | Speedup |
|--------------|---------|
| Larger batch (384 vs 256) | 1.5x |
| More workers (32 vs 16) | 1.5x |
| torch.compile() | 1.2x |
| Channels-last | 1.15x |
| TF32 | 1.1x |
| No mixup/cutmix | 1.1x |
| **Total** | **~3.5x** |

### Realistic Expectations

- **First epoch:** Slow (torch.compile compilation)
- **Epochs 2-5:** Warming up (data loading pipeline)
- **Epoch 6+:** Full speed (~1.5-2.5 it/s)

## Validation

After 1-2 epochs, you should see:

```
Epoch 1: 100%|██████████| 3336/3336 [45:23<00:00,  1.22it/s]
Validation: 100%|██████████| 196/196 [02:15<00:00,  1.45it/s]

Epoch 1 metrics:
  train/loss: 6.234
  train/acc1: 0.8%
  val/loss: 6.123
  val/acc1: 1.2%
  val/acc5: 5.4%
```

**Success indicators:**
- ✓ Training speed: 1.0-2.5 it/s
- ✓ GPU utilization: 70-95%
- ✓ Time per epoch: 30-60 minutes
- ✓ No OOM errors

## Next Steps

1. **Run 1-2 epochs** to confirm performance
2. **If GPU util is high (>80%):** Re-enable mixup/cutmix
3. **If still low (<50%):** Investigate FSx or system bottlenecks
4. **If OOM:** Reduce batch size to 320 or 256

## Summary

**Main changes:**
- Batch size: 256 → 384
- Workers: 16 → 32
- Prefetch: 4 → 8
- Added: torch.compile, channels_last, TF32
- Disabled: mixup/cutmix (temporarily)

**Expected result:**
- **~10-15x faster** than original (0.19 it/s → 2.0 it/s)
- **~2-4 days** for 100 epochs (was 29 days)
- **GPU utilization: 70-95%** (was 0-10%)
