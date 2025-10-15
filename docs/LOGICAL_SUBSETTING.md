# Logical Subsetting Guide

## Overview

The DataModule now supports **logical subsetting** - you can use a subset of the full dataset without creating separate directories. This is perfect for:
- Quick smoke tests on GPU instances
- Testing with S3-mounted data
- Experimenting with different subset sizes
- Avoiding the need to create physical subset directories

---

## How It Works

Instead of creating `/data/imagenet-tiny`, you use the full dataset and specify subset parameters:

```python
dm = ImageNetDataModule(
    data_root="/data/imagenet",      # Full dataset
    max_classes=10,                   # Only use 10 classes
    max_samples_per_class=100,        # Only 100 samples per class
)
```

The DataModule will:
1. Load the full dataset structure
2. Randomly select `max_classes` classes (reproducible with `subset_seed`)
3. Randomly select `max_samples_per_class` samples from each class
4. Create a PyTorch `Subset` with only those samples

---

## Configuration Examples

### Tiny Subset (Quick Smoke Test)
```yaml
# configs/tiny_gpu.yaml
data_root: /mnt/s3/imagenet
max_classes: 10
max_samples_per_class: 100
batch_size: 128
epochs: 5
```

**Result**: 1,000 train samples, 100 val samples, ~5 minutes per epoch

### Medium Subset (Faster Experiments)
```yaml
# configs/medium.yaml
data_root: /mnt/s3/imagenet
max_classes: 100
max_samples_per_class: 500
batch_size: 256
epochs: 30
```

**Result**: 50,000 train samples, 1,000 val samples, ~30 minutes per epoch

### Full Dataset (Production Training)
```yaml
# configs/full.yaml
data_root: /mnt/s3/imagenet
# max_classes: null  # Use all 1000 classes
# max_samples_per_class: null  # Use all samples
batch_size: 256
epochs: 90
```

**Result**: 1.28M train samples, 50K val samples

---

## Usage

### Command Line

```bash
# Tiny subset
python train.py --config configs/tiny_gpu.yaml

# Or override via command line
python train.py \
    --config configs/base.yaml \
    --data_root /mnt/s3/imagenet \
    --max_classes 10 \
    --max_samples_per_class 100 \
    --epochs 5

# Medium subset
python train.py \
    --config configs/base.yaml \
    --max_classes 100 \
    --max_samples_per_class 500

# Full dataset (no subsetting)
python train.py --config configs/base.yaml
```

### Python API

```python
from data.datamodule import ImageNetDataModule

# Tiny subset
dm = ImageNetDataModule(
    data_root="/mnt/s3/imagenet",
    max_classes=10,
    max_samples_per_class=100,
    batch_size=128,
    num_workers=8
)

# Medium subset
dm = ImageNetDataModule(
    data_root="/mnt/s3/imagenet",
    max_classes=100,
    max_samples_per_class=500,
    batch_size=256,
    num_workers=8
)

# Full dataset
dm = ImageNetDataModule(
    data_root="/mnt/s3/imagenet",
    # No max_classes or max_samples_per_class
    batch_size=256,
    num_workers=8
)
```

---

## Parameters

### `max_classes` (Optional[int])
- **Default**: `None` (use all 1000 classes)
- **Description**: Maximum number of classes to include
- **Example**: `max_classes=10` → use only 10 randomly selected classes

### `max_samples_per_class` (Optional[int])
- **Default**: `None` (use all samples)
- **Description**: Maximum training samples per class
- **Note**: Validation uses 10 samples per class when this is set
- **Example**: `max_samples_per_class=100` → 100 train samples per class

### `subset_seed` (int)
- **Default**: `42`
- **Description**: Random seed for reproducible subset selection
- **Note**: Same seed = same classes and samples selected

---

## GPU Testing Workflow

### Step 1: Quick Smoke Test (5-10 minutes)
```bash
# Test with tiny subset first
python train.py \
    --config configs/tiny_gpu.yaml \
    --data_root /mnt/s3/imagenet \
    --epochs 1
```

**Verify:**
- ✅ Data loads from S3
- ✅ GPU training works
- ✅ No errors in pipeline

### Step 2: Small Training Run (30-60 minutes)
```bash
# Train for a few epochs to verify learning
python train.py \
    --config configs/tiny_gpu.yaml \
    --data_root /mnt/s3/imagenet \
    --epochs 10
```

**Verify:**
- ✅ Training loss decreases
- ✅ Validation accuracy improves
- ✅ Checkpointing works

### Step 3: Full Training
```bash
# Switch to full dataset
python train.py \
    --config configs/local.yaml \
    --data_root /mnt/s3/imagenet \
    --epochs 90
```

---

## Advantages Over Physical Subsets

### Physical Subset (Old Approach)
```bash
# Create subset directory
python scripts/make_tiny_subset.py \
    --source /data/imagenet \
    --target /data/imagenet-tiny \
    --num_classes 10

# Use subset
python train.py --data_root /data/imagenet-tiny
```

**Cons:**
- ❌ Extra step to create subset
- ❌ Disk space for symlinks/copies
- ❌ Need to recreate for different sizes
- ❌ Extra directories to manage
- ❌ Not available on new machines

### Logical Subset (New Approach)
```bash
# Just run with parameters
python train.py \
    --data_root /data/imagenet \
    --max_classes 10 \
    --max_samples_per_class 100
```

**Pros:**
- ✅ No extra files
- ✅ Change size via config
- ✅ Works immediately on any machine
- ✅ Simpler workflow
- ✅ Perfect for S3/cloud storage

---

## Performance

### Subset Creation Time
- **Physical subset**: 5-10 seconds (create symlinks)
- **Logical subset**: 1-2 seconds (iterate directory once)

### Training Performance
- **No difference** - both use the same underlying data
- Logical subset has negligible overhead (~1-2 seconds at startup)

---

## Examples for Different Use Cases

### Quick Pipeline Test
```yaml
max_classes: 5
max_samples_per_class: 50
epochs: 1
# ~250 samples, completes in minutes
```

### Hyperparameter Tuning
```yaml
max_classes: 50
max_samples_per_class: 200
epochs: 20
# ~10K samples, fast iterations
```

### Ablation Studies
```yaml
max_classes: 100
max_samples_per_class: 500
epochs: 30
# ~50K samples, meaningful results
```

### Full Training
```yaml
# No subsetting parameters
epochs: 90
# Full 1.28M samples
```

---

## Reproducibility

The subset selection is **deterministic** based on `subset_seed`:

```python
# These will select the SAME classes and samples
dm1 = ImageNetDataModule(max_classes=10, subset_seed=42)
dm2 = ImageNetDataModule(max_classes=10, subset_seed=42)

# These will select DIFFERENT classes and samples
dm3 = ImageNetDataModule(max_classes=10, subset_seed=123)
```

---

## Testing

Run the test suite to verify logical subsetting:

```bash
python tests/test_logical_subset.py
```

**Tests:**
- ✅ Small subset (5 classes, 50 samples)
- ✅ Medium subset (10 classes, 100 samples)
- ✅ Class-only limiting (3 classes, all samples)
- ✅ Full dataset (no subsetting)
- ✅ Dataloader compatibility

---

## Migration from Physical Subsets

If you have existing code using physical subsets:

### Before
```yaml
data_root: /data/imagenet-tiny
num_classes: 10
```

### After
```yaml
data_root: /data/imagenet
max_classes: 10
max_samples_per_class: 100
num_classes: 10
```

**Note**: Make sure `num_classes` matches `max_classes` in your config!

---

## FAQ

### Q: Does this slow down training?
**A:** No. The subset is created once at startup (1-2 seconds). Training speed is identical.

### Q: Can I use this with S3/cloud storage?
**A:** Yes! This is perfect for S3. No need to create subset directories.

### Q: Is the subset selection random?
**A:** Yes, but reproducible. Use `subset_seed` to get the same subset every time.

### Q: Can I limit only classes or only samples?
**A:** Yes! Set only `max_classes` or only `max_samples_per_class`.

### Q: What happens to validation data?
**A:** When `max_samples_per_class` is set, validation uses 10 samples per class.

### Q: Can I see which classes were selected?
**A:** The DataModule prints: `"Created subset: X samples from Y classes"`

---

## Recommended Workflow for GPU Instance

```bash
# 1. Clone repo
git clone <repo> && cd imagenet

# 2. Setup environment
conda create -n imagenet python=3.11 -y
conda activate imagenet
pip install -r requirements.txt

# 3. Mount S3 data (or verify data path)
ls /mnt/s3/imagenet/train

# 4. Quick test with tiny subset
python train.py \
    --config configs/tiny_gpu.yaml \
    --data_root /mnt/s3/imagenet \
    --epochs 1 \
    --no_wandb

# 5. If successful, run full training
python train.py \
    --config configs/local.yaml \
    --data_root /mnt/s3/imagenet \
    --epochs 90
```

---

## Summary

**Logical subsetting** provides a simpler, more flexible way to work with dataset subsets:

- ✅ No physical subset directories needed
- ✅ Change subset size via config
- ✅ Perfect for S3/cloud storage
- ✅ Reproducible with seed
- ✅ No performance penalty
- ✅ Cleaner workflow

**Use it for**: Quick tests, hyperparameter tuning, ablation studies, and GPU smoke tests before full training runs.
