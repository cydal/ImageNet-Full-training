#!/usr/bin/env python3
"""
Debug script to check if train and val labels are properly aligned.
"""
import torch
from torchvision import datasets
from pathlib import Path

data_root = Path("/mnt/nvme_data/imagenet_subset")

# Load datasets
train_dataset = datasets.ImageFolder(root=data_root / "train")
val_dataset = datasets.ImageFolder(root=data_root / "val")

print("="*70)
print("LABEL ALIGNMENT CHECK")
print("="*70)

print(f"\nTrain dataset:")
print(f"  Classes: {len(train_dataset.classes)}")
print(f"  Samples: {len(train_dataset)}")
print(f"  First 10 classes: {train_dataset.classes[:10]}")
print(f"  Class to idx (first 10): {dict(list(train_dataset.class_to_idx.items())[:10])}")

print(f"\nVal dataset:")
print(f"  Classes: {len(val_dataset.classes)}")
print(f"  Samples: {len(val_dataset)}")
print(f"  First 10 classes: {val_dataset.classes[:10]}")
print(f"  Class to idx (first 10): {dict(list(val_dataset.class_to_idx.items())[:10])}")

# Check if class mappings are identical
if train_dataset.class_to_idx == val_dataset.class_to_idx:
    print("\n✅ Class mappings are IDENTICAL")
else:
    print("\n❌ Class mappings are DIFFERENT!")
    print("\nDifferences:")
    for cls in train_dataset.classes:
        if cls in val_dataset.class_to_idx:
            if train_dataset.class_to_idx[cls] != val_dataset.class_to_idx[cls]:
                print(f"  {cls}: train={train_dataset.class_to_idx[cls]}, val={val_dataset.class_to_idx[cls]}")
        else:
            print(f"  {cls}: in train but NOT in val")
    
    for cls in val_dataset.classes:
        if cls not in train_dataset.class_to_idx:
            print(f"  {cls}: in val but NOT in train")

# Sample a few items to check labels
print("\n" + "="*70)
print("SAMPLE LABELS")
print("="*70)

print("\nTrain samples (first 5):")
for i in range(min(5, len(train_dataset))):
    img, label = train_dataset[i]
    class_name = train_dataset.classes[label]
    print(f"  Sample {i}: label={label}, class={class_name}")

print("\nVal samples (first 5):")
for i in range(min(5, len(val_dataset))):
    img, label = val_dataset[i]
    class_name = val_dataset.classes[label]
    print(f"  Sample {i}: label={label}, class={class_name}")

print("\n" + "="*70)
