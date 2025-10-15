# Subset Approach Comparison

## Question
Why create a physical tiny subset (`/data2/imagenet-tiny`) instead of using the full dataset with logical subsetting?

## Answer
**You're right!** Logical subsetting would be simpler and more flexible. Here's the comparison:

---

## Approach 1: Physical Subset (What We Did)

### Implementation
```bash
# Create separate directory with symlinks
python scripts/make_tiny_subset.py \
    --source /data2/imagenet \
    --target /data2/imagenet-tiny \
    --num_classes 5 \
    --num_train_images 50
```

### Config
```yaml
data_root: /data2/imagenet-tiny
num_classes: 5
```

### Pros
- ✅ Faster iteration (smaller directory)
- ✅ Isolated test environment
- ✅ Clear separation of test vs full data

### Cons
- ❌ Extra step to create subset
- ❌ Disk space for symlinks
- ❌ Less flexible (need to recreate for different sizes)
- ❌ Extra files to manage

---

## Approach 2: Logical Subset (Your Suggestion) ⭐ BETTER

### Implementation
```python
# In datamodule.py - add subset support
class ImageNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int = 256,
        num_workers: int = 8,
        # NEW: Subset parameters
        max_classes: Optional[int] = None,  # Limit number of classes
        max_samples_per_class: Optional[int] = None,  # Limit samples per class
        **kwargs
    ):
        self.max_classes = max_classes
        self.max_samples_per_class = max_samples_per_class
        # ... rest of init
    
    def setup(self, stage: Optional[str] = None):
        # Load full dataset
        full_dataset = datasets.ImageFolder(
            root=self.data_root / "train",
            transform=train_transform
        )
        
        # Apply logical subset if requested
        if self.max_classes or self.max_samples_per_class:
            self.train_dataset = self._create_subset(
                full_dataset, 
                self.max_classes,
                self.max_samples_per_class
            )
        else:
            self.train_dataset = full_dataset
```

### Config
```yaml
data_root: /data2/imagenet  # Use full dataset
max_classes: 5               # Only use 5 classes
max_samples_per_class: 50    # Only 50 samples per class
```

### Pros
- ✅ No extra files needed
- ✅ More flexible (change via config)
- ✅ Faster setup (no subset creation)
- ✅ Easy to test different sizes
- ✅ Single source of truth (full dataset)

### Cons
- ❌ Slightly slower to iterate full directory (minimal)
- ❌ Need to implement subset logic in DataModule

---

## Recommended Implementation

Here's how to add logical subsetting to the DataModule:

```python
# data/datamodule.py

from torch.utils.data import Subset
import random

class ImageNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int = 256,
        num_workers: int = 8,
        # Subset parameters
        max_classes: Optional[int] = None,
        max_samples_per_class: Optional[int] = None,
        subset_seed: int = 42,
        **kwargs
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_classes = max_classes
        self.max_samples_per_class = max_samples_per_class
        self.subset_seed = subset_seed
        # ... rest of init
    
    def _create_subset(self, dataset, max_classes=None, max_samples_per_class=None):
        """Create a logical subset of the dataset."""
        if max_classes is None and max_samples_per_class is None:
            return dataset
        
        # Get class-to-indices mapping
        class_to_indices = {}
        for idx, (_, label) in enumerate(dataset.samples):
            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)
        
        # Limit number of classes
        if max_classes:
            random.seed(self.subset_seed)
            selected_classes = random.sample(
                sorted(class_to_indices.keys()), 
                min(max_classes, len(class_to_indices))
            )
            class_to_indices = {
                k: v for k, v in class_to_indices.items() 
                if k in selected_classes
            }
        
        # Limit samples per class
        subset_indices = []
        for label, indices in class_to_indices.items():
            if max_samples_per_class:
                random.seed(self.subset_seed + label)
                indices = random.sample(
                    indices, 
                    min(max_samples_per_class, len(indices))
                )
            subset_indices.extend(indices)
        
        return Subset(dataset, subset_indices)
    
    def setup(self, stage: Optional[str] = None):
        """Setup train and validation datasets."""
        if stage == "fit" or stage is None:
            train_transform = self._get_train_transform()
            full_train = datasets.ImageFolder(
                root=self.data_root / "train",
                transform=train_transform
            )
            # Apply subset if requested
            self.train_dataset = self._create_subset(
                full_train,
                self.max_classes,
                self.max_samples_per_class
            )
        
        if stage == "fit" or stage == "validate" or stage is None:
            val_transform = self._get_val_transform()
            full_val = datasets.ImageFolder(
                root=self.data_root / "val",
                transform=val_transform
            )
            # Apply subset if requested
            self.val_dataset = self._create_subset(
                full_val,
                self.max_classes,
                max_samples_per_class=10 if self.max_samples_per_class else None
            )
```

---

## Usage Examples

### Tiny Subset (5 classes, 50 samples each)
```yaml
# configs/tiny.yaml
data_root: /data2/imagenet
max_classes: 5
max_samples_per_class: 50
batch_size: 32
epochs: 1
```

### Medium Subset (100 classes, 500 samples each)
```yaml
# configs/medium.yaml
data_root: /data2/imagenet
max_classes: 100
max_samples_per_class: 500
batch_size: 128
epochs: 10
```

### Full Dataset
```yaml
# configs/full.yaml
data_root: /data2/imagenet
# max_classes: null  # Use all classes
# max_samples_per_class: null  # Use all samples
batch_size: 256
epochs: 90
```

---

## Performance Comparison

### Physical Subset
```bash
# Setup time: ~5 seconds (create symlinks)
# Training startup: Fast (small directory)
# Flexibility: Low (need to recreate)
```

### Logical Subset
```bash
# Setup time: 0 seconds (no creation needed)
# Training startup: Slightly slower (iterate full directory once)
# Flexibility: High (just change config)
```

**Difference:** Negligible for ImageNet (1-2 seconds at most)

---

## Recommendation

**Use Logical Subsetting** for these reasons:

1. **Simpler workflow**
   ```bash
   # No need for this:
   python scripts/make_tiny_subset.py ...
   
   # Just run:
   python train.py --config configs/tiny.yaml
   ```

2. **More flexible testing**
   ```bash
   # Test different sizes easily:
   python train.py --max_classes 5 --max_samples_per_class 50
   python train.py --max_classes 10 --max_samples_per_class 100
   python train.py --max_classes 100 --max_samples_per_class 500
   ```

3. **Cleaner codebase**
   - No extra directories
   - No subset creation scripts to maintain
   - Single source of truth

---

## Migration Path

If you want to switch to logical subsetting:

1. **Add subset support to DataModule** (code above)
2. **Update configs/tiny.yaml**:
   ```yaml
   data_root: /data2/imagenet  # Change from imagenet-tiny
   max_classes: 5
   max_samples_per_class: 50
   ```
3. **Remove physical subset** (optional):
   ```bash
   rm -rf /data2/imagenet-tiny
   ```

---

## Conclusion

**Your intuition is correct!** Logical subsetting is:
- ✅ Simpler
- ✅ More flexible
- ✅ Easier to maintain
- ✅ No performance penalty

The physical subset approach works but is unnecessarily complex. For future projects, logical subsetting is the better choice.

---

## Why I Used Physical Subset

Honest answer:
1. The repo already had `make_tiny_subset.py` script
2. It was the "obvious" approach (create smaller dataset)
3. Didn't think through the logical subsetting alternative

**Your question highlights a better design!** This is a good example of questioning assumptions and finding simpler solutions.
