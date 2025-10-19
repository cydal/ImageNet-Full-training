#!/usr/bin/env python3
"""
Check if train and val datasets have the same label mapping when loaded through datamodule.
"""
import yaml
from data.datamodule import ImageNetDataModule

# Load config
with open("configs/pretrained_test.yaml") as f:
    config = yaml.safe_load(f)

# Create datamodule
dm = ImageNetDataModule(**config)
dm.setup(stage="fit")

print("="*70)
print("DATAMODULE LABEL MAPPING CHECK")
print("="*70)

# Check train dataset
print("\nTrain dataset:")
print(f"  Type: {type(dm.train_dataset)}")
print(f"  Length: {len(dm.train_dataset)}")

if hasattr(dm.train_dataset, 'label_mapping'):
    print(f"  Has label_mapping: Yes")
    print(f"  Label mapping (first 10): {dict(list(dm.train_dataset.label_mapping.items())[:10])}")
else:
    print(f"  Has label_mapping: No")

# Check val dataset
print("\nVal dataset:")
print(f"  Type: {type(dm.val_dataset)}")
print(f"  Length: {len(dm.val_dataset)}")

if hasattr(dm.val_dataset, 'label_mapping'):
    print(f"  Has label_mapping: Yes")
    print(f"  Label mapping (first 10): {dict(list(dm.val_dataset.label_mapping.items())[:10])}")
else:
    print(f"  Has label_mapping: No")

# Sample a few items
print("\n" + "="*70)
print("SAMPLE ITEMS")
print("="*70)

print("\nTrain samples (first 5):")
for i in range(5):
    img, label = dm.train_dataset[i]
    print(f"  Sample {i}: label={label}")

print("\nVal samples (first 5):")
for i in range(5):
    img, label = dm.val_dataset[i]
    print(f"  Sample {i}: label={label}")

# Check if mappings are the same
if hasattr(dm.train_dataset, 'label_mapping') and hasattr(dm.val_dataset, 'label_mapping'):
    if dm.train_dataset.label_mapping == dm.val_dataset.label_mapping:
        print("\n✅ Label mappings are IDENTICAL")
    else:
        print("\n❌ Label mappings are DIFFERENT!")
        print("\nThis is the BUG!")
