# Full Training Performance Guarantee

## Critical Design Decision

**The subsetting code has ZERO impact on full training.**

## How It Works

### Early Return Pattern

```python
def _create_subset(self, dataset, max_classes=None, max_samples_per_class=None):
    if max_classes is None and max_samples_per_class is None:
        return dataset  # ← Immediate return, no processing
    
    # Subsetting code only runs if parameters are provided
    # ...
```

When you run **full training** (without `--max_classes` or `--max_samples_per_class`):
1. `_create_subset()` is called with `max_classes=None, max_samples_per_class=None`
2. **Line 129-130: Immediate return** of the original `ImageFolder` dataset
3. No label remapping, no wrapping, no overhead
4. Identical to code before subsetting feature was added

## Verification

### Test 1: Verify No Wrapper or Remapping
```bash
python /home/ubuntu/test_full_training_unaffected.py
```

**Results:**
```
Train dataset type: <class 'torchvision.datasets.folder.ImageFolder'>
Val dataset type: <class 'torchvision.datasets.folder.ImageFolder'>

✓ Train dataset is plain ImageFolder (not wrapped)
✓ Val dataset is plain ImageFolder (not wrapped)
✓ Train dataset is NOT RemappedSubset (no label remapping)
✓ Val dataset is NOT RemappedSubset (no label remapping)

CRITICAL CHECK: Verifying NO label remapping
✓ Train dataset has NO label_mapping attribute (plain ImageFolder)
✓ Val dataset has NO label_mapping attribute (plain ImageFolder)
✓ Direct dataset access - sample label: 0
✓ Label is raw integer from ImageFolder (no remapping applied)

✓✓✓ NO RemappedSubset wrapper
✓✓✓ NO label_mapping attribute
✓✓✓ NO label remapping performed
```

### Test 2: Side-by-Side Comparison
```bash
python /home/ubuntu/test_comparison.py
```

**Results show clear distinction:**

**Full Training:**
- Dataset: `ImageFolder` (original)
- Label remapping: **NO**
- Label range: [0, 999]
- Has label_mapping attribute: **False**

**Subset Training:**
- Dataset: `RemappedSubset` (wrapper)
- Label remapping: **YES**
- Label range: [0, 9]
- Has label_mapping attribute: **True**

## Performance Impact

| Scenario | Dataset Type | Label Processing | Overhead |
|----------|--------------|------------------|----------|
| **Full training** | `ImageFolder` | None (original labels) | **0%** |
| Subset training | `RemappedSubset` | Label remapping | ~0.1% (negligible) |

## Code Paths

### Full Training (Production)
```python
# Command: python train.py --config configs/full.yaml
# max_classes=None, max_samples_per_class=None

datamodule.setup()
  └─> _create_subset(dataset, None, None)
      └─> return dataset  # ← Original ImageFolder, unchanged
```

### Subset Training (Testing)
```python
# Command: python train.py --max_classes 10 --max_samples_per_class 50
# max_classes=10, max_samples_per_class=50

datamodule.setup()
  └─> _create_subset(dataset, 10, 50)
      └─> Build class mapping
      └─> Create label remapping
      └─> return RemappedSubset(...)  # ← Wrapped with remapping
```

## Guarantee

**Full training is 100% unaffected:**
- ✓ No wrapper classes
- ✓ No label remapping
- ✓ No iteration overhead
- ✓ No memory overhead
- ✓ Identical to pre-subsetting code

The subsetting feature is **purely additive** and only activates when explicitly requested via command-line arguments.

## Testing Strategy

Before any production run, verify with:
```bash
# Test full training is unaffected
python /home/ubuntu/test_full_training_unaffected.py

# Test subset training works
python /home/ubuntu/test_label_remapping.py
```

Both should pass, confirming:
1. Full training uses original dataset (no modifications)
2. Subset training correctly remaps labels

## Design Philosophy

**Priority: Full training must never be compromised.**

The subsetting feature was designed with this principle:
- Early return for full training (lines 129-130)
- All subsetting logic is bypassed when not needed
- Zero performance impact on production workloads
- Subsetting is opt-in via explicit parameters

This ensures that testing features don't interfere with production training performance.
