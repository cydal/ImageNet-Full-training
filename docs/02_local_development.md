# Local Development Guide

## Setup

### 1. Verify Data Access
Your ImageNet data is located at `/data2/imagenet`:
```bash
ls /data2/imagenet
# Should show: train/ val/

# Check number of classes
ls /data2/imagenet/train | wc -l
# Should show: 1000

# Check a sample class
ls /data2/imagenet/train/n01440764 | head -5
```

### 2. Install Dependencies
```bash
cd /home/ubuntu/imagenet
make install
```

### 3. Test Data Module
```bash
make test-data
```

This will:
- Load the data module with local path `/data2/imagenet`
- Verify dataset sizes (train: ~1.28M, val: 50K)
- Test data loading speed
- Check image preprocessing
- Benchmark throughput

## Configuration

### Local Development Config
Use `configs/local.yaml` for local development:
```yaml
data_root: /data2/imagenet
batch_size: 128
epochs: 10
num_workers: 8
```

### Key Differences from Production
| Setting | Local | Production (FSx) |
|---------|-------|------------------|
| data_root | `/data2/imagenet` | `/fsx/imagenet` |
| batch_size | 128 | 256 |
| epochs | 10 | 90 |
| num_workers | 8 | 12 |
| use_ema | false | true |

## Testing Workflow

### 1. Test Data Module
```bash
# Quick pipeline test
make quick-test

# Comprehensive data module test
make test-data

# Data integrity test
make test-integrity

# Benchmark dataloader
make benchmark-data

# Run all tests
make test-all
```

Expected output:
- Train dataset: ~1,281,167 images
- Val dataset: 50,000 images
- Number of classes: 1000
- Batch shape: [64, 3, 224, 224]
- Throughput: 500-2000 images/sec (depends on hardware)

### 2. Test Model Forward Pass
```bash
python -c "
from models.resnet50 import ResNet50Module
import torch

model = ResNet50Module(num_classes=1000)
x = torch.randn(4, 3, 224, 224)
y = model(x)
print(f'Output shape: {y.shape}')
print(f'Expected: torch.Size([4, 1000])')
"
```

### 3. Quick Training Test (1 epoch, 100 batches)
```bash
python train.py \
    --config configs/local.yaml \
    --epochs 1 \
    --batch_size 64 \
    --no_wandb
```

### 4. Full Local Training
```bash
make train-local
```

## Development Checklist

### Phase 1: Data Module ✓
- [x] Verify data structure
- [x] Create local config
- [x] Test data loading
- [x] Benchmark throughput
- [ ] Optimize num_workers

### Phase 2: Model Testing
- [ ] Test forward pass
- [ ] Test loss computation
- [ ] Test backward pass
- [ ] Verify gradient flow

### Phase 3: Training Loop
- [ ] Single GPU training (1 epoch)
- [ ] Verify metrics logging
- [ ] Test checkpointing
- [ ] Test resuming from checkpoint

### Phase 4: Multi-GPU
- [ ] Test DDP on single node
- [ ] Verify gradient synchronization
- [ ] Test batch norm synchronization
- [ ] Benchmark scaling efficiency

### Phase 5: Production Ready
- [ ] Switch to FSx data
- [ ] Test multi-node setup
- [ ] Full training run (90 epochs)
- [ ] Validate final accuracy

## Common Issues

### Issue: Permission denied on /data2/imagenet
```bash
# Check permissions
ls -la /data2/imagenet

# If needed, adjust permissions (be careful!)
sudo chown -R ubuntu:ubuntu /data2/imagenet
```

### Issue: Slow data loading
**Symptoms**: Low GPU utilization, slow epoch time

**Solutions**:
1. Increase `num_workers` (try 8, 12, or 16)
2. Enable `persistent_workers: true`
3. Check disk I/O with `iostat -x 1`
4. Consider using faster storage or caching

### Issue: Out of memory
**Symptoms**: CUDA out of memory error

**Solutions**:
1. Reduce `batch_size` (try 64, 32)
2. Reduce `num_workers`
3. Use gradient accumulation
4. Check GPU memory: `nvidia-smi`

### Issue: Workers timing out
**Symptoms**: DataLoader worker timeout errors

**Solutions**:
1. Reduce `num_workers`
2. Increase timeout in DataLoader
3. Check for corrupted images

## Performance Tuning

### Optimal num_workers
Rule of thumb: 4-12 workers per GPU
```bash
# Test different values
for nw in 4 8 12 16; do
    echo "Testing num_workers=$nw"
    python test_datamodule.py --num_workers $nw
done
```

### Batch Size Selection
- **Small GPU (8-16GB)**: batch_size=64-128
- **Medium GPU (16-24GB)**: batch_size=128-256
- **Large GPU (24GB+)**: batch_size=256-512

### Data Loading Bottleneck Check
```python
import time
from data.datamodule import ImageNetDataModule

dm = ImageNetDataModule(data_root="/data2/imagenet", batch_size=256, num_workers=8)
dm.setup("fit")
loader = dm.train_dataloader()

# Measure data loading time
times = []
for i, batch in enumerate(loader):
    start = time.time()
    # Simulate GPU processing
    time.sleep(0.05)  # 50ms per batch
    times.append(time.time() - start)
    if i >= 100:
        break

print(f"Avg time per batch: {sum(times)/len(times):.3f}s")
print(f"If > 0.1s, data loading is a bottleneck")
```

## Next Steps
1. Run `make test-data` to verify data module
2. Test model forward pass
3. Run single epoch training test
4. Proceed to multi-GPU testing

## Useful Commands

```bash
# Check GPU status
nvidia-smi

# Monitor GPU utilization
watch -n 1 nvidia-smi

# Check disk I/O
iostat -x 1

# Monitor training
tail -f logs/csv_logs/version_0/metrics.csv

# Check checkpoint size
du -sh checkpoints/
```
