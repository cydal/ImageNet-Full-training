#!/usr/bin/env python3
"""
Test logical subsetting functionality.
Verifies that max_classes and max_samples_per_class work correctly.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.datamodule import ImageNetDataModule


def test_logical_subset():
    """Test logical subsetting with different configurations."""
    print("=" * 70)
    print("Testing Logical Subsetting")
    print("=" * 70)
    
    # Test 1: Small subset (5 classes, 50 samples each)
    print("\n[Test 1] Small subset: 5 classes, 50 samples per class")
    dm = ImageNetDataModule(
        data_root="/data2/imagenet",
        max_classes=5,
        max_samples_per_class=50,
        batch_size=32,
        num_workers=2
    )
    dm.setup("fit")
    
    print(f"  Train dataset size: {len(dm.train_dataset)}")
    print(f"  Val dataset size: {len(dm.val_dataset)}")
    print(f"  Expected train: 5 × 50 = 250")
    print(f"  Expected val: 5 × 10 = 50")
    
    assert len(dm.train_dataset) == 250, f"Expected 250 train samples, got {len(dm.train_dataset)}"
    assert len(dm.val_dataset) == 50, f"Expected 50 val samples, got {len(dm.val_dataset)}"
    print("  ✓ Test 1 passed!")
    
    # Test 2: Medium subset (10 classes, 100 samples each)
    print("\n[Test 2] Medium subset: 10 classes, 100 samples per class")
    dm = ImageNetDataModule(
        data_root="/data2/imagenet",
        max_classes=10,
        max_samples_per_class=100,
        batch_size=64,
        num_workers=4
    )
    dm.setup("fit")
    
    print(f"  Train dataset size: {len(dm.train_dataset)}")
    print(f"  Val dataset size: {len(dm.val_dataset)}")
    print(f"  Expected train: 10 × 100 = 1000")
    print(f"  Expected val: 10 × 10 = 100")
    
    assert len(dm.train_dataset) == 1000, f"Expected 1000 train samples, got {len(dm.train_dataset)}"
    assert len(dm.val_dataset) == 100, f"Expected 100 val samples, got {len(dm.val_dataset)}"
    print("  ✓ Test 2 passed!")
    
    # Test 3: Only limit classes (all samples)
    print("\n[Test 3] Limit classes only: 3 classes, all samples")
    dm = ImageNetDataModule(
        data_root="/data2/imagenet",
        max_classes=3,
        max_samples_per_class=None,  # Use all samples
        batch_size=32,
        num_workers=2
    )
    dm.setup("fit")
    
    print(f"  Train dataset size: {len(dm.train_dataset)}")
    print(f"  Val dataset size: {len(dm.val_dataset)}")
    print(f"  Expected train: ~3800 (3 classes × ~1281 avg)")
    print(f"  Expected val: 150 (3 classes × 50)")
    
    # Should be around 3800 for 3 classes
    assert 3500 < len(dm.train_dataset) < 4000, f"Unexpected train size: {len(dm.train_dataset)}"
    assert len(dm.val_dataset) == 150, f"Expected 150 val samples, got {len(dm.val_dataset)}"
    print("  ✓ Test 3 passed!")
    
    # Test 4: No subsetting (full dataset)
    print("\n[Test 4] No subsetting: full dataset")
    dm = ImageNetDataModule(
        data_root="/data2/imagenet",
        max_classes=None,
        max_samples_per_class=None,
        batch_size=256,
        num_workers=8
    )
    dm.setup("fit")
    
    print(f"  Train dataset size: {len(dm.train_dataset)}")
    print(f"  Val dataset size: {len(dm.val_dataset)}")
    print(f"  Expected train: ~1,281,167")
    print(f"  Expected val: 50,000")
    
    assert len(dm.train_dataset) == 1281167, f"Expected 1281167 train samples, got {len(dm.train_dataset)}"
    assert len(dm.val_dataset) == 50000, f"Expected 50000 val samples, got {len(dm.val_dataset)}"
    print("  ✓ Test 4 passed!")
    
    # Test 5: Verify dataloader works
    print("\n[Test 5] Verify dataloader works with subset")
    dm = ImageNetDataModule(
        data_root="/data2/imagenet",
        max_classes=5,
        max_samples_per_class=50,
        batch_size=32,
        num_workers=2
    )
    dm.setup("fit")
    
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    
    # Get one batch
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    
    print(f"  Train batch shape: {train_batch[0].shape}")
    print(f"  Val batch shape: {val_batch[0].shape}")
    print(f"  Train labels range: {train_batch[1].min()}-{train_batch[1].max()}")
    print(f"  Val labels range: {val_batch[1].min()}-{val_batch[1].max()}")
    
    assert train_batch[0].shape[0] <= 32, "Batch size too large"
    assert train_batch[0].shape[1:] == (3, 224, 224), "Wrong image shape"
    print("  ✓ Test 5 passed!")
    
    print("\n" + "=" * 70)
    print("✅ All logical subsetting tests passed!")
    print("=" * 70)
    print("\n📝 Summary:")
    print("  - Logical subsetting works correctly")
    print("  - No need to create physical subset directories")
    print("  - Can easily change subset size via config")
    print("  - Ready for GPU testing with S3 data")


if __name__ == "__main__":
    test_logical_subset()
