# Getting Started with Local Development

## Prerequisites
- Ubuntu Linux
- NVIDIA GPU with CUDA support
- ImageNet dataset at `/data2/imagenet`
- Python 3.8+

## Quick Start (5 minutes)

### 1. Install Dependencies
```bash
cd /home/ubuntu/imagenet
make install
```

This installs:
- PyTorch 2.0+
- PyTorch Lightning 2.0+
- torchvision, timm
- wandb for logging

### 2. Run Quick Test
```bash
make quick-test
```

This verifies:
- ✓ Data loading from `/data2/imagenet`
- ✓ Model creation (ResNet50)
- ✓ Forward pass
- ✓ Loss computation
- ✓ Backward pass and gradients

**Expected output:**
```
[1/4] Testing Data Module...
  ✓ Train samples: 1,281,167
  ✓ Val samples: 50,000

[2/4] Testing Model Creation...
  ✓ Model created: ResNet50Module
  ✓ Parameters: 25,557,032

[3/4] Testing Forward Pass...
  ✓ Batch shape: torch.Size([32, 3, 224, 224])
  ✓ Output shape: torch.Size([32, 1000])

[4/4] Testing Training Step...
  ✓ Loss computed: 6.9077
  ✓ Backward pass successful

✅ All tests passed!
```

### 3. Test Data Module (Optional)
```bash
make test-data
```

This runs comprehensive data loading tests:
- Dataset statistics
- Data loading speed benchmark
- Augmentation verification
- Class distribution check

### 4. Run Single Epoch Training
```bash
python train.py --config configs/local.yaml --epochs 1 --no_wandb
```

This trains for 1 epoch to verify the full pipeline.

**Expected behavior:**
- Loads data from `/data2/imagenet`
- Trains ResNet50 from scratch
- Logs metrics every 20 steps
- Saves checkpoint after epoch
- Takes ~30-60 minutes (depending on hardware)

### 5. Full Local Training
```bash
make train-local
```

This runs full training with local config:
- 10 epochs (vs 90 for production)
- Batch size 128
- Learning rate 0.05
- No EMA (for faster training)

## Understanding the Configuration

### Local Config (`configs/local.yaml`)
```yaml
data_root: /data2/imagenet    # Local data path
batch_size: 128               # Adjust for your GPU
epochs: 10                    # Quick training
lr: 0.05                      # Scaled with batch size
num_workers: 8                # Adjust for your CPU
```

### Adjusting for Your Hardware

**Small GPU (8-16GB VRAM):**
```bash
python train.py --config configs/local.yaml --batch_size 64
```

**Large GPU (24GB+ VRAM):**
```bash
python train.py --config configs/local.yaml --batch_size 256
```

**Multiple GPUs:**
```bash
python train.py --config configs/local.yaml --devices 4
```

## Monitoring Training

### Real-time Logs
```bash
# Watch training progress
tail -f logs/csv_logs/version_0/metrics.csv

# Monitor GPU
watch -n 1 nvidia-smi
```

### Weights & Biases (Optional)
```bash
# Login to W&B
wandb login

# Train with W&B logging
python train.py --config configs/local.yaml
```

### Checkpoints
Checkpoints are saved to `checkpoints/`:
```
checkpoints/
├── resnet50-epoch=00-val_acc1=0.1234.ckpt
├── resnet50-epoch=01-val_acc1=0.2345.ckpt
└── last.ckpt
```

## Common Workflows

### Development Iteration
```bash
# 1. Make code changes
vim models/resnet50.py

# 2. Quick test
make quick-test

# 3. Single epoch test
python train.py --config configs/local.yaml --epochs 1 --no_wandb

# 4. Full training
make train-local
```

### Debugging
```bash
# Test with smaller batch
python train.py --config configs/local.yaml --batch_size 32 --epochs 1

# Test with fewer workers
python train.py --config configs/local.yaml --num_workers 2 --epochs 1

# Test single GPU
python train.py --config configs/local.yaml --devices 1 --epochs 1
```

### Evaluation
```bash
# Evaluate best checkpoint
make eval CHECKPOINT=checkpoints/resnet50-epoch=09-val_acc1=0.7123.ckpt

# Or directly
python eval.py --checkpoint checkpoints/last.ckpt --config configs/local.yaml
```

## Troubleshooting

### Issue: "No module named 'pytorch_lightning'"
```bash
make install
```

### Issue: "FileNotFoundError: /data2/imagenet"
```bash
# Check if data exists
ls /data2/imagenet

# Check if mounted
df -h /data2

# Mount if needed
sudo mount /dev/xvdf /data2
```

### Issue: CUDA out of memory
```bash
# Reduce batch size
python train.py --config configs/local.yaml --batch_size 64

# Or use gradient accumulation (coming soon)
```

### Issue: Slow data loading
```bash
# Increase workers
python train.py --config configs/local.yaml --num_workers 12

# Or test different values
for nw in 4 8 12 16; do
    echo "Testing $nw workers"
    python test_datamodule.py
done
```

## Next Steps

1. ✅ Verify quick test passes
2. ✅ Run single epoch training
3. ⏳ Run full local training (10 epochs)
4. ⏳ Test multi-GPU on single node
5. ⏳ Prepare for FSx migration
6. ⏳ Setup multi-node training

## Performance Expectations

### Single GPU (A100 40GB)
- **Throughput**: ~800-1200 images/sec
- **Epoch time**: ~20-30 minutes
- **10 epochs**: ~3-5 hours

### 4x GPUs (A100 40GB)
- **Throughput**: ~3000-4000 images/sec
- **Epoch time**: ~5-8 minutes
- **10 epochs**: ~1 hour

### 8x GPUs (A100 40GB)
- **Throughput**: ~5000-7000 images/sec
- **Epoch time**: ~3-5 minutes
- **10 epochs**: ~30-45 minutes

## Useful Commands

```bash
# Show all available commands
make help

# Quick pipeline test
make quick-test

# Test data module
make test-data

# Train locally
make train-local

# Clean up
make clean

# Check GPU
nvidia-smi

# Check disk usage
df -h /data2

# Check data
ls /data2/imagenet/train | wc -l  # Should be 1000
```

## Documentation

- `00_project_overview.md` - Project goals and structure
- `01_data_module.md` - Data loading details
- `02_local_development.md` - Development guide
- `03_getting_started.md` - This file
