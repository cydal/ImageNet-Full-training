#!/usr/bin/env python
"""Test that num_classes is correctly determined from dataset."""

from data.datamodule import ImageNetDataModule

# Test with subset
print("Testing with max_classes=10, max_samples_per_class=50")
dm = ImageNetDataModule(
    data_root="/fsx/ns1",
    batch_size=32,
    max_classes=10,
    max_samples_per_class=50
)

# Setup to initialize datasets
dm.setup(stage="fit")

# Get number of classes
num_classes = dm.num_classes()
print(f"✓ Number of classes: {num_classes}")

if num_classes == 10:
    print("✓ SUCCESS: Correct number of classes!")
else:
    print(f"✗ FAIL: Expected 10 classes, got {num_classes}")
    exit(1)

# Test with full dataset
print("\nTesting with full dataset (no subsetting)")
dm_full = ImageNetDataModule(
    data_root="/fsx/ns1",
    batch_size=32
)
dm_full.setup(stage="fit")
num_classes_full = dm_full.num_classes()
print(f"✓ Number of classes: {num_classes_full}")

if num_classes_full == 1000:
    print("✓ SUCCESS: Full ImageNet has 1000 classes!")
else:
    print(f"Note: Dataset has {num_classes_full} classes (not standard 1000)")
