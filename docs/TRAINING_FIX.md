# Training Error Fix: Class Mismatch

## The Problem

When running:
```bash
python train.py --config configs/tiny_gpu.yaml --epochs 1 --no_wandb --max_classes 10 --max_samples_per_class 50
```

You got:
```
torch.AcceleratorError: CUDA error: device-side assert triggered
/pytorch/aten/src/ATen/native/cuda/Loss.cu:245: nll_loss_forward_reduce_cuda_kernel_2d: 
block: [0,0,0], thread: [0,0,0] Assertion `t >= 0 && t < n_classes` failed.
```

## Root Cause

**Two issues**:

1. **Class count mismatch**: The model was initialized with `num_classes=1000` (default ImageNet), but the dataset only had 10 classes (due to `--max_classes 10`).

2. **Label remapping missing**: When subsetting ImageNet classes, the original class labels (e.g., 437, 891) were preserved, but the model expected labels in range [0, 9]. PyTorch's loss function failed with assertion error because labels were outside the valid range.

## The Fix

### 1. Created `RemappedSubset` class for label remapping

```python
class RemappedSubset(Dataset):
    """Subset with remapped labels to [0, N-1]."""
    def __init__(self, dataset, indices, label_mapping=None):
        self.dataset = dataset
        self.indices = indices
        self.label_mapping = label_mapping  # original_label -> new_label
    
    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        if self.label_mapping is not None:
            label = self.label_mapping[label]  # Remap to [0, N-1]
        return image, label
```

This ensures labels are always in range [0, num_classes-1].

### 2. Added `num_classes()` method to `ImageNetDataModule`

```python
def num_classes(self) -> int:
    """Get the number of classes in the dataset."""
    return self._num_classes
```

Tracks the actual number of classes after subsetting.

### 3. Updated `train.py` to use correct number of classes

```python
# Create data module
datamodule = ImageNetDataModule(**config)

# Setup datamodule to determine number of classes
datamodule.setup(stage="fit")

# Override num_classes in config based on actual dataset
config["num_classes"] = datamodule.num_classes()
print(f"Using {config['num_classes']} classes for training")

# Create model with correct number of classes
model = ResNet50Module(**config)
```

Now the model's final layer is correctly sized for the actual number of classes in the dataset.

## How to Use

### Subset Training (Testing)

```bash
# Train on 10 classes with 50 samples each
python train.py \
    --config configs/tiny_gpu.yaml \
    --epochs 1 \
    --no_wandb \
    --max_classes 10 \
    --max_samples_per_class 50
```

The model will automatically use `num_classes=10`.

### Full Training

```bash
# Train on all 1000 ImageNet classes
python train.py --config configs/full.yaml
```

The model will automatically use `num_classes=1000`.

## Testing the Fix

Run the test script:
```bash
python test_num_classes.py
```

This verifies that:
- Subset datasets correctly report their number of classes
- Full datasets report 1000 classes
- The `num_classes()` method works correctly

## Additional Notes

### Missing Dependency

If you get `ModuleNotFoundError: No module named 'yaml'`, install it:
```bash
conda activate imagenet
pip install pyyaml
```

### Label Remapping

**Important**: When using `max_classes`, the subset uses a **random sample** of ImageNet classes. The original class indices (e.g., n01440764 = tench) are preserved in the dataset, but PyTorch's `ImageFolder` automatically remaps them to consecutive indices [0, N-1] for the subset.

This means:
- Original ImageNet class 437 might become class 0 in your subset
- The model trains on classes [0, 9] for a 10-class subset
- Predictions are in the subset's class space, not original ImageNet

If you need to map back to original ImageNet classes, use:
```python
# Get class mapping
class_to_idx = datamodule.train_dataset.dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}
```

## Files Modified

1. **`data/datamodule.py`**:
   - Added `RemappedSubset` class for label remapping
   - Added `_num_classes` tracking
   - Added `num_classes()` method
   - Updated `_create_subset()` to create label mapping and use `RemappedSubset`
   - Updated `setup()` to initialize `_num_classes`

2. **`train.py`**:
   - Call `datamodule.setup()` before creating model
   - Set `config["num_classes"]` from `datamodule.num_classes()`
   - Print number of classes being used

## Testing

Run the test to verify label remapping works:
```bash
python /home/ubuntu/test_label_remapping.py
```

Expected output:
```
✓ Number of classes: 10
✓ Label values in batch: [3, 4, 7, 8]
✓ Min label: 3, Max label: 8
✓ SUCCESS: All labels are in range [0, 9]
```
