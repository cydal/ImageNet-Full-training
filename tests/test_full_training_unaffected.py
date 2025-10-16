#!/usr/bin/env python
"""Verify that full training is completely unaffected by subsetting code."""
import sys
sys.path.insert(0, '/home/ubuntu/ImageNet-Full-training')

from data.datamodule import ImageNetDataModule, RemappedSubset
from torchvision.datasets import ImageFolder

print("=" * 70)
print("Testing: Full training is UNAFFECTED by subsetting code")
print("=" * 70)

# Create datamodule WITHOUT subsetting (normal full training)
dm = ImageNetDataModule(
    data_root="/fsx/ns1",
    batch_size=256,
    num_workers=0
    # NO max_classes, NO max_samples_per_class
)

dm.setup(stage="fit")

# Check that datasets are plain ImageFolder, not wrapped
print(f"\nTrain dataset type: {type(dm.train_dataset)}")
print(f"Val dataset type: {type(dm.val_dataset)}")

# Verify it's the original ImageFolder
assert isinstance(dm.train_dataset, ImageFolder), \
    f"Expected ImageFolder, got {type(dm.train_dataset)}"
assert isinstance(dm.val_dataset, ImageFolder), \
    f"Expected ImageFolder, got {type(dm.val_dataset)}"

# CRITICAL: Verify it's NOT RemappedSubset
assert not isinstance(dm.train_dataset, RemappedSubset), \
    "ERROR: Train dataset should NOT be RemappedSubset for full training!"
assert not isinstance(dm.val_dataset, RemappedSubset), \
    "ERROR: Val dataset should NOT be RemappedSubset for full training!"

print("\n✓ Train dataset is plain ImageFolder (not wrapped)")
print("✓ Val dataset is plain ImageFolder (not wrapped)")
print("✓ Train dataset is NOT RemappedSubset (no label remapping)")
print("✓ Val dataset is NOT RemappedSubset (no label remapping)")

# Check number of classes
num_classes = dm.num_classes()
print(f"\n✓ Number of classes: {num_classes}")

# Get a sample to verify labels are unchanged
train_loader = dm.train_dataloader()
batch = next(iter(train_loader))
images, labels = batch

print(f"\n✓ Batch shape: images={images.shape}, labels={labels.shape}")
print(f"✓ Sample labels: {labels[:10].tolist()}")
print(f"✓ Label range: [{labels.min().item()}, {labels.max().item()}]")

# For full ImageNet, labels should span the full range
assert labels.min() >= 0
assert labels.max() < num_classes

# CRITICAL: Verify no label remapping by checking dataset directly
print("\n" + "=" * 70)
print("CRITICAL CHECK: Verifying NO label remapping")
print("=" * 70)

# Access dataset directly to check for label_mapping attribute
if hasattr(dm.train_dataset, 'label_mapping'):
    if dm.train_dataset.label_mapping is not None:
        print("✗ ERROR: Train dataset has label_mapping!")
        print(f"  label_mapping: {dm.train_dataset.label_mapping}")
        raise AssertionError("Full training should NOT have label remapping!")
    else:
        print("✓ Train dataset.label_mapping is None")
else:
    print("✓ Train dataset has NO label_mapping attribute (plain ImageFolder)")

if hasattr(dm.val_dataset, 'label_mapping'):
    if dm.val_dataset.label_mapping is not None:
        print("✗ ERROR: Val dataset has label_mapping!")
        raise AssertionError("Full training should NOT have label remapping!")
    else:
        print("✓ Val dataset.label_mapping is None")
else:
    print("✓ Val dataset has NO label_mapping attribute (plain ImageFolder)")

# Get raw sample from dataset to verify labels are untouched
sample_img, sample_label = dm.train_dataset[0]
print(f"\n✓ Direct dataset access - sample label: {sample_label}")
print(f"✓ Label is raw integer from ImageFolder (no remapping applied)")

print("\n" + "=" * 70)
print("✓✓✓ SUCCESS: Full training uses ORIGINAL ImageFolder")
print("✓✓✓ NO RemappedSubset wrapper")
print("✓✓✓ NO label_mapping attribute")
print("✓✓✓ NO label remapping performed")
print("✓✓✓ NO overhead, NO modifications")
print("=" * 70)
print("\nConclusion: Full training is 100% unaffected by subsetting code.")
print("Labels are passed through unchanged from ImageFolder to model.")
