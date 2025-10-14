# GPU Migration Guide

**Status**: Ready to move to GPU machine  
**Date**: October 14, 2025

---

## Summary

All validation and smoke tests completed successfully on CPU machine. Implementation is verified and ready for GPU training.

---

## What's Been Validated ✅

### 1. Data Pipeline
- ✅ Dataset integrity (0% corruption, 1.28M train, 50K val)
- ✅ Data loading working correctly
- ✅ Preprocessing and augmentation functional
- ✅ All 1000 classes present and valid

### 2. Training Implementation
- ✅ End-to-end pipeline functional
- ✅ Model creation and forward pass
- ✅ Loss computation and backward pass
- ✅ Optimizer and learning rate scheduler
- ✅ Validation and metrics logging
- ✅ Checkpointing system

### 3. Configuration System
- ✅ YAML-based configs working
- ✅ Config inheritance (base → specific)
- ✅ Command-line overrides
- ✅ Flexible hardware settings (CPU/GPU)

---

## Migration Checklist

### On Current (CPU) Machine

```bash
# 1. Commit all changes
cd /home/ubuntu/imagenet
git add .
git commit -m "CPU validation complete - ready for GPU"
git push

# 2. Note the commit hash
git log -1 --oneline
```

### On GPU Machine

```bash
# 1. Clone repository
git clone https://github.com/cydal/ImageNet-Full-training.git imagenet
cd imagenet

# 2. Setup environment
conda create -n imagenet python=3.11 -y
conda activate imagenet
pip install -r requirements.txt

# 3. Verify GPU
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# 4. Setup data path
# Option A: If data is already on GPU machine
ls /path/to/imagenet

# Option B: If need to copy/mount data
# (depends on your setup - FSx, local copy, etc.)

# 5. Update config for GPU
# Edit configs/local.yaml or create configs/gpu.yaml
```

---

## Configuration Changes for GPU

### Create `configs/gpu.yaml`

```yaml
# GPU configuration for single-node training
# Inherits from base.yaml

# Data settings
data_root: /path/to/imagenet  # Update this path
batch_size: 256               # Increase for GPU
num_workers: 8                # Increase for GPU

# Hardware settings
accelerator: gpu
devices: 1                    # Single GPU
strategy: auto                # Or 'ddp' for multi-GPU
precision: 16-mixed           # Mixed precision for speed
sync_batchnorm: false         # Only needed for multi-GPU

# Training settings
epochs: 90                    # Full training
lr: 0.1                       # Standard ImageNet LR
warmup_epochs: 5
max_epochs: 90

# Augmentation (enable for better accuracy)
auto_augment: imagenet        # Or null for baseline
mixup_alpha: 0.2              # Or 0.0 to disable
cutmix_alpha: 1.0             # Or 0.0 to disable
label_smoothing: 0.1          # Or 0.0 to disable

# Logging
log_every_n_steps: 50
val_check_interval: 1.0       # Validate after each epoch

# Checkpointing
save_top_k: 3
monitor: val/acc1
mode: max
```

### Or Update `configs/local.yaml`

```yaml
# Change these settings:
accelerator: gpu              # Was: cpu
devices: 1                    # Or more for multi-GPU
precision: 16-mixed           # Was: 32
batch_size: 256               # Was: 128
num_workers: 8                # Was: 8 (may increase to 12)
epochs: 90                    # Was: 10
```

---

## First GPU Test (Recommended)

### Quick Validation (1 epoch, ~30 min)

```bash
# Activate environment
conda activate imagenet

# Run 1 epoch to verify GPU training
python train.py \
    --config configs/local.yaml \
    --accelerator gpu \
    --devices 1 \
    --precision 16-mixed \
    --epochs 1 \
    --no_wandb
```

**Expected results:**
- Training completes without errors
- GPU utilization high (80-100%)
- Throughput: 800-1200 images/sec
- Epoch time: 20-30 minutes
- Validation accuracy: 30-40% (after 1 epoch)

---

## Full Training Run

### Single GPU

```bash
# With W&B logging
python train.py \
    --config configs/local.yaml \
    --accelerator gpu \
    --devices 1 \
    --precision 16-mixed \
    --epochs 90 \
    --wandb_project imagenet-training \
    --wandb_name resnet50-baseline

# Without W&B
python train.py \
    --config configs/local.yaml \
    --accelerator gpu \
    --devices 1 \
    --precision 16-mixed \
    --epochs 90 \
    --no_wandb
```

**Expected:**
- Total time: 30-45 hours (single A100)
- Final top-1 accuracy: ~76%
- Final top-5 accuracy: ~93%

### Multi-GPU (if available)

```bash
python train.py \
    --config configs/local.yaml \
    --accelerator gpu \
    --devices 4 \
    --strategy ddp \
    --precision 16-mixed \
    --epochs 90
```

**Expected:**
- 4x speedup (approximately)
- Epoch time: 5-8 minutes
- Total time: 8-12 hours

---

## Monitoring Training

### Real-time Monitoring

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Watch training logs
tail -f logs/csv_logs/version_*/metrics.csv

# Monitor checkpoints
ls -lh checkpoints/
```

### Weights & Biases (Recommended)

```bash
# Login (one time)
wandb login

# Training will automatically log to W&B
# View at: https://wandb.ai/your-username/imagenet-training
```

### Check Progress

```bash
# View latest metrics
tail -20 logs/csv_logs/version_*/metrics.csv

# Check best checkpoint
ls -lh checkpoints/ | grep "val_acc1"
```

---

## Performance Expectations

### Single GPU (A100 40GB)

| Metric | Expected Value |
|--------|----------------|
| Throughput | 800-1200 images/sec |
| Batch size | 256-512 |
| Epoch time | 20-30 minutes |
| 90 epochs | 30-45 hours |
| GPU memory | 15-25 GB |
| GPU utilization | 80-100% |

### Multi-GPU (4x A100)

| Metric | Expected Value |
|--------|----------------|
| Throughput | 3000-4000 images/sec |
| Batch size | 256 per GPU (1024 total) |
| Epoch time | 5-8 minutes |
| 90 epochs | 8-12 hours |
| Scaling efficiency | 85-95% |

---

## Troubleshooting

### Issue: CUDA out of memory

**Solutions:**
```bash
# Reduce batch size
python train.py --config configs/local.yaml --batch_size 128

# Or use gradient accumulation (future feature)
```

### Issue: Low GPU utilization

**Possible causes:**
1. Data loading bottleneck
   - Increase `num_workers` (try 12, 16)
   - Use faster storage (NVMe, FSx)
   
2. Small batch size
   - Increase `batch_size` (try 384, 512)
   
3. CPU bottleneck
   - Check with `htop`
   - Increase `num_workers`

### Issue: Training slower than expected

**Check:**
```bash
# 1. GPU utilization
nvidia-smi

# 2. Data loading speed
# Run: python tests/benchmark_dataloader.py

# 3. Disk I/O
iostat -x 1

# 4. CPU usage
htop
```

---

## Optimization Tips

### 1. Batch Size Tuning

```bash
# Find maximum batch size that fits in GPU memory
for bs in 256 384 512 768; do
    echo "Testing batch_size=$bs"
    python train.py --config configs/local.yaml --batch_size $bs --epochs 1
done
```

### 2. num_workers Tuning

```bash
# Test different worker counts
python tests/benchmark_dataloader.py --test workers
```

### 3. Mixed Precision

```bash
# Always use mixed precision on modern GPUs
--precision 16-mixed  # Recommended for A100, V100, etc.
```

### 4. Data Loading

```bash
# If data loading is slow:
- Increase num_workers (8, 12, 16)
- Enable persistent_workers: true
- Use faster storage (NVMe SSD, FSx)
- Consider DALI for maximum speed (future)
```

---

## Expected Training Timeline

### Day 1: Setup & Validation (2-3 hours)
- Clone repo and setup environment
- Verify GPU and data access
- Run 1-epoch test
- Verify metrics and checkpointing

### Day 2-3: Full Training (30-45 hours)
- Start 90-epoch training
- Monitor progress
- Check metrics periodically

### Day 4: Evaluation
- Training completes
- Evaluate final model
- Analyze results

---

## Success Criteria

### After 1 Epoch
- ✅ Training completes without errors
- ✅ GPU utilization > 80%
- ✅ Throughput > 800 images/sec
- ✅ Validation accuracy > 30%

### After 90 Epochs
- ✅ Training completes successfully
- ✅ Top-1 accuracy: 75-77%
- ✅ Top-5 accuracy: 92-94%
- ✅ Checkpoints saved correctly

---

## Files to Transfer

**Essential:**
- All code (via git)
- `requirements.txt`
- `configs/` directory

**Data:**
- ImageNet dataset (if not already on GPU machine)
- Or setup FSx mount

**Not needed:**
- Tiny subset (`/data2/imagenet-tiny`)
- Test logs
- CPU checkpoints

---

## Quick Start Commands

```bash
# On GPU machine:

# 1. Setup
git clone <repo> imagenet && cd imagenet
conda create -n imagenet python=3.11 -y
conda activate imagenet
pip install -r requirements.txt

# 2. Verify
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# 3. Test (1 epoch)
python train.py --config configs/local.yaml --epochs 1 --no_wandb

# 4. Full training
python train.py --config configs/local.yaml --epochs 90
```

---

## Contact & Support

If issues arise:
1. Check `docs/02_local_development.md` for troubleshooting
2. Review error messages in logs
3. Check GPU memory with `nvidia-smi`
4. Verify data path is correct

---

**Status**: ✅ Ready for GPU Migration  
**Confidence**: HIGH - All components validated  
**Next Step**: Clone repo on GPU machine and run 1-epoch test
