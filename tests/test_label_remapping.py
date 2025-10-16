#!/usr/bin/env python
"""Test that label remapping works correctly."""
import sys
sys.path.insert(0, '/home/ubuntu/ImageNet-Full-training')

from data.datamodule import ImageNetDataModule
import torch

print("Testing label remapping with max_classes=10, max_samples_per_class=50")
print("=" * 60)

dm = ImageNetDataModule(
    data_root="/fsx/ns1",
    batch_size=4,
    num_workers=0,  # Avoid multiprocessing for testing
    max_classes=10,
    max_samples_per_class=50
)

# Setup
dm.setup(stage="fit")

print(f"\n✓ Number of classes: {dm.num_classes()}")

# Get a batch from train dataloader
train_loader = dm.train_dataloader()
batch = next(iter(train_loader))
images, labels = batch

print(f"\n✓ Batch shape: images={images.shape}, labels={labels.shape}")
print(f"✓ Label values in batch: {labels.tolist()}")
print(f"✓ Min label: {labels.min().item()}, Max label: {labels.max().item()}")

# Verify labels are in correct range
assert labels.min() >= 0, f"Labels should be >= 0, got {labels.min()}"
assert labels.max() < dm.num_classes(), f"Labels should be < {dm.num_classes()}, got {labels.max()}"

print(f"\n✓ SUCCESS: All labels are in range [0, {dm.num_classes()-1}]")

# Test validation set too
val_loader = dm.val_dataloader()
val_batch = next(iter(val_loader))
val_images, val_labels = val_batch

print(f"\n✓ Val batch: images={val_images.shape}, labels={val_labels.shape}")
print(f"✓ Val label values: {val_labels.tolist()}")
print(f"✓ Val min label: {val_labels.min().item()}, max label: {val_labels.max().item()}")

assert val_labels.min() >= 0
assert val_labels.max() < dm.num_classes()

print(f"\n✓ SUCCESS: Validation labels also in correct range!")
print("\n" + "=" * 60)
print("✓✓✓ ALL TESTS PASSED ✓✓✓")
print("=" * 60)
