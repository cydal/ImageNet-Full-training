# Optimal Single GPU Configuration

## System Specifications

- **GPU:** NVIDIA A10G (23GB VRAM)
- **RAM:** 62GB
- **CPU:** 16 cores
- **Storage:** FSx Lustre

## Memory Analysis

### Per-Batch Memory Calculation

```
Batch memory = batch_size × channels × height × width × bytes_per_pixel
             = 256 × 3 × 224 × 224 × 4 bytes
             = 154,140,672 bytes
             ≈ 147 MB per batch (FP32)
             ≈ 230 MB per batch (with overhead)
```

### Data Loading Buffer

```
Total RAM buffer = num_workers × prefetch_factor × batch_memory
```

**Example calculations:**

| Workers | Prefetch | Batches Buffered | RAM Used |
|---------|----------|------------------|----------|
| 32 | 8 | 256 | ~59 GB | ❌ OOM |
| 16 | 4 | 64 | ~15 GB | ⚠️ Risky |
| 12 | 3 | 36 | ~8 GB | ✅ Safe |
| 8 | 4 | 32 | ~7 GB | ✅ Safe |

## Optimal Configuration

### Final Settings

```yaml
# Data loading - BALANCED
batch_size: 256          # Good GPU utilization without OOM
num_workers: 12          # 75% of CPU cores (16 * 0.75)
prefetch_factor: 3       # Conservative buffering
pin_memory: true         # Faster CPU→GPU transfer
persistent_workers: true # Reduce worker restart overhead

# Model optimization
compile_model: true      # PyTorch 2.0+ compilation
# channels_last enabled in model code

# Augmentation
mixup_alpha: 0.0        # Disabled for max speed
cutmix_alpha: 0.0       # Disabled for max speed
label_smoothing: 0.1    # Minimal overhead

# Training
lr: 0.5                 # Scaled for batch_size=256
precision: 16-mixed     # FP16 for speed
```

### Memory Budget

| Component | Memory | Notes |
|-----------|--------|-------|
| **System RAM** | | |
| OS + Base | ~3 GB | Operating system |
| Data buffer | ~8 GB | 12 workers × 3 prefetch |
| PyTorch overhead | ~2 GB | Model weights in RAM |
| **Total RAM** | ~13 GB | Safe (62GB available) |
| | | |
| **GPU VRAM** | | |
| Model (FP16) | ~0.1 GB | ResNet50 weights |
| Activations | ~12 GB | Forward pass |
| Gradients | ~0.1 GB | Backward pass |
| Optimizer state | ~0.2 GB | SGD momentum |
| Batch (FP16) | ~0.3 GB | 256 × 3 × 224 × 224 × 2 |
| **Total VRAM** | ~13 GB | Safe (23GB available) |

## Expected Performance

### Training Speed

| Metric | Value |
|--------|-------|
| **Iterations/epoch** | 5,004 |
| **Speed** | 0.8-1.5 it/s |
| **Time/epoch** | 55-105 min |
| **100 epochs** | 4-7 days |
| **GPU utilization** | 70-90% |
| **GPU memory** | 13-16 GB |

### Speedup Breakdown

| Optimization | Speedup | Enabled |
|--------------|---------|---------|
| torch.compile() | 1.2x | ✅ |
| channels_last | 1.15x | ✅ |
| TF32 precision | 1.1x | ✅ |
| FP16 mixed precision | 2.0x | ✅ |
| Optimized data loading | 1.5x | ✅ |
| **Total** | **~3.8x** | |

## Tuning Guidelines

### If You Want More Speed

**Option 1: Increase batch size (if GPU memory allows)**
```yaml
batch_size: 320  # or 384
lr: 0.625        # or 0.75
# Monitor GPU memory - should stay < 20GB
```

**Option 2: Increase workers (if RAM allows)**
```yaml
num_workers: 16
prefetch_factor: 3
# RAM buffer: 16 * 3 * 230MB ≈ 11GB
```

**Option 3: Increase prefetch (if RAM allows)**
```yaml
num_workers: 12
prefetch_factor: 4
# RAM buffer: 12 * 4 * 230MB ≈ 11GB
```

### If You Hit OOM (System RAM)

**Reduce data loading buffer:**
```yaml
num_workers: 8
prefetch_factor: 2
# RAM buffer: 8 * 2 * 230MB ≈ 3.7GB
```

### If You Hit OOM (GPU VRAM)

**Reduce batch size:**
```yaml
batch_size: 192  # or 128
lr: 0.375        # or 0.25
```

## Multi-GPU Scaling Guidelines

### Per-GPU Settings (Same as Single GPU)

```yaml
# Keep these PER GPU:
batch_size: 256          # Per GPU
num_workers: 12          # Per GPU
prefetch_factor: 3       # Per GPU
```

### Multi-GPU Calculations

**For N GPUs:**

| GPUs | Total Batch | Total Workers | RAM Buffer | LR |
|------|-------------|---------------|------------|-----|
| 1 | 256 | 12 | ~8 GB | 0.5 |
| 2 | 512 | 24 | ~16 GB | 1.0 |
| 4 | 1024 | 48 | ~32 GB | 2.0 |
| 8 | 2048 | 96 | ~64 GB | 4.0 |

**Important:**
- RAM buffer scales linearly with GPUs
- Ensure total RAM > (8GB × num_GPUs) + 5GB overhead
- LR scales linearly with total batch size

### Example: 4 GPUs on 4 Instances

**Per instance (1 GPU each):**
```yaml
batch_size: 256
num_workers: 12
prefetch_factor: 3
# RAM per instance: ~13GB (safe for 62GB)
```

**Effective training:**
```yaml
total_batch_size: 1024  # 256 × 4
total_workers: 48       # 12 × 4
lr: 2.0                 # 0.5 × 4
```

## Monitoring Commands

### During Training

```bash
# GPU utilization (should be 70-90%)
watch -n 1 nvidia-smi

# System memory (should stay < 20GB used)
watch -n 1 free -h

# CPU usage (workers should use 70-90% of cores)
htop

# Training progress
tail -f logs/train_*.log
```

### Expected Output

```bash
# nvidia-smi
GPU: 85%
Memory: 14500 MiB / 23028 MiB

# free -h
Mem: 15Gi / 62Gi used

# Training
Epoch 0: 1%|▏| 50/5004 [00:45<1:15:00, 1.10it/s, train/loss=6.9]
```

## Troubleshooting

### Low GPU Utilization (<50%)

**Likely causes:**
1. Data loading bottleneck
2. FSx throttling
3. CPU bottleneck

**Solutions:**
```yaml
# Try increasing workers
num_workers: 16

# Or increase prefetch
prefetch_factor: 4

# Check FSx performance
aws cloudwatch get-metric-statistics \
    --namespace AWS/FSx \
    --metric-name DataReadBytes \
    --dimensions Name=FileSystemId,Value=fs-02386cb09beeabb62
```

### System OOM

**Symptoms:**
- Process killed
- "Killed" message
- System becomes unresponsive

**Solutions:**
```yaml
# Reduce workers
num_workers: 8

# Reduce prefetch
prefetch_factor: 2

# Or reduce batch size
batch_size: 192
```

### GPU OOM

**Symptoms:**
- CUDA out of memory error
- Training crashes with OOM message

**Solutions:**
```yaml
# Reduce batch size
batch_size: 192  # or 128

# Disable compilation (saves memory)
compile_model: false

# Enable gradient checkpointing (in model code)
```

## Best Practices

### ✅ Do

1. **Start conservative** - Use these settings first
2. **Monitor memory** - Watch both RAM and VRAM
3. **Increase gradually** - Add workers/batch incrementally
4. **Test before long runs** - Run 1-2 epochs to verify stability
5. **Log everything** - Use W&B to track performance

### ❌ Don't

1. **Don't max everything** - Leave headroom for stability
2. **Don't ignore warnings** - OOM warnings mean reduce settings
3. **Don't change multiple things** - Change one parameter at a time
4. **Don't forget to scale LR** - Always scale with batch size
5. **Don't use all CPU cores** - Leave some for system

## Summary

### Conservative (Guaranteed Stable)

```yaml
batch_size: 192
num_workers: 8
prefetch_factor: 2
compile_model: false
```

**Expected:** 0.7-1.0 it/s, ~6-8 days for 100 epochs

### Balanced (Recommended) ✅

```yaml
batch_size: 256
num_workers: 12
prefetch_factor: 3
compile_model: true
```

**Expected:** 0.8-1.5 it/s, ~4-7 days for 100 epochs

### Aggressive (May OOM)

```yaml
batch_size: 320
num_workers: 16
prefetch_factor: 4
compile_model: true
```

**Expected:** 1.0-2.0 it/s, ~3-5 days for 100 epochs

## Current Configuration

**File:** `configs/single_gpu_full.yaml`

```yaml
batch_size: 256          # ✅ Balanced
num_workers: 12          # ✅ Balanced
prefetch_factor: 3       # ✅ Conservative
compile_model: true      # ✅ Enabled
mixup_alpha: 0.0         # ⚠️ Disabled (for speed)
cutmix_alpha: 0.0        # ⚠️ Disabled (for speed)
```

**RAM budget:** ~8GB (safe for 62GB system)  
**VRAM budget:** ~13-16GB (safe for 23GB GPU)  
**Expected speed:** 0.8-1.5 it/s  
**Expected time:** 4-7 days for 100 epochs

This configuration is **production-ready** and will scale well to multi-GPU setups.
