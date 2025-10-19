# Local ImageNet Subset Information

## Copy Status

**Copying in progress...**

- **Source**: `/mnt/s3-imagenet/imagenet` (S3 mount)
- **Destination**: `/mnt/nvme_data/imagenet_subset` (NVMe local disk)
- **Classes**: 100 (selected with seed=42)
- **Train samples**: 500 per class = 50,000 total
- **Val samples**: 10 per class = 1,000 total
- **Estimated size**: ~6.57 GB
- **Estimated time**: ~30 minutes

## Selected Classes (First 10)

```
n01498041, n01629819, n01631663, n01641577, n01644900
n01689811, n01693334, n01770393, n01795545, n01796340
```

## After Copy Completes

### Update Your Configs

For any config file you want to use with local data, update:

```yaml
data_root: /mnt/nvme_data/imagenet_subset
```

### Configs to Update

1. **`configs/stress_test.yaml`** - Already has subset parameters
2. **`configs/optimized_test.yaml`** - For aggressive prefetching tests
3. **`configs/single_gpu_full.yaml`** - If you want to test with local data

### Expected Performance Improvement

**Before (S3 mount)**:
- Epoch 0: 0.43 it/s (very slow, cold cache)
- Epoch 1: 1.42 it/s (warming up)
- Epoch 2: 1.60 it/s (better)
- GPU utilization: Inconsistent (0-100%)

**After (Local NVMe)**:
- All epochs: 3-5 it/s (consistent)
- GPU utilization: 90-100% (consistent)
- **5-10x faster overall**

### Test Command (After Copy)

```bash
# Quick test with local data
conda activate imagenet && python train.py \
  --config configs/stress_test.yaml \
  --data_root /mnt/nvme_data/imagenet_subset \
  --no_wandb \
  --epochs 3 \
  --batch_size 256
```

### Verify Copy

After copy completes, verify:

```bash
# Check directory structure
ls -lh /mnt/nvme_data/imagenet_subset/

# Count classes
ls /mnt/nvme_data/imagenet_subset/train/ | wc -l  # Should be 100
ls /mnt/nvme_data/imagenet_subset/val/ | wc -l    # Should be 100

# Check total size
du -sh /mnt/nvme_data/imagenet_subset/
```

## Disk Space

- **NVMe disk**: 521 GB available
- **Subset size**: ~6.57 GB
- **Remaining after copy**: ~514 GB
- **Safe for**: Multiple experiments, checkpoints, logs

## Notes

- The subset uses the **same random seed (42)** as the datamodule
- Classes are selected deterministically
- This matches exactly what the logical subsetting would select
- No need to specify `--max_classes` or `--max_samples_per_class` when using this local copy
- The data is already filtered to the subset
