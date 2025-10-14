#!/usr/bin/env python3
"""
Test script for ImageNet DataModule.
Verifies data loading, preprocessing, and basic statistics.
"""
import time
import torch
from data.datamodule import ImageNetDataModule


def test_datamodule():
    """Test the ImageNet data module."""
    print("=" * 60)
    print("Testing ImageNet DataModule")
    print("=" * 60)
    
    # Create data module with local path
    print("\n1. Creating DataModule...")
    dm = ImageNetDataModule(
        data_root="/data2/imagenet",
        batch_size=64,
        num_workers=4,
        img_size=224,
        pin_memory=True,
        persistent_workers=True
    )
    print("✓ DataModule created")
    
    # Setup datasets
    print("\n2. Setting up datasets...")
    dm.setup("fit")
    print(f"✓ Train dataset: {len(dm.train_dataset):,} images")
    print(f"✓ Val dataset: {len(dm.val_dataset):,} images")
    print(f"✓ Number of classes: {len(dm.train_dataset.classes)}")
    
    # Test train dataloader
    print("\n3. Testing train dataloader...")
    train_loader = dm.train_dataloader()
    print(f"✓ Number of batches: {len(train_loader):,}")
    
    # Get first batch
    start_time = time.time()
    batch = next(iter(train_loader))
    load_time = time.time() - start_time
    
    images, labels = batch
    print(f"✓ Batch loaded in {load_time:.3f}s")
    print(f"  - Images shape: {images.shape}")
    print(f"  - Labels shape: {labels.shape}")
    print(f"  - Images dtype: {images.dtype}")
    print(f"  - Images range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  - Labels range: [{labels.min()}, {labels.max()}]")
    
    # Test validation dataloader
    print("\n4. Testing validation dataloader...")
    val_loader = dm.val_dataloader()
    print(f"✓ Number of batches: {len(val_loader):,}")
    
    # Get first batch
    start_time = time.time()
    batch = next(iter(val_loader))
    load_time = time.time() - start_time
    
    images, labels = batch
    print(f"✓ Batch loaded in {load_time:.3f}s")
    print(f"  - Images shape: {images.shape}")
    print(f"  - Labels shape: {labels.shape}")
    
    # Benchmark data loading speed
    print("\n5. Benchmarking data loading speed...")
    num_batches = 50
    start_time = time.time()
    
    for i, batch in enumerate(train_loader):
        if i >= num_batches:
            break
    
    elapsed = time.time() - start_time
    images_per_sec = (num_batches * dm.batch_size) / elapsed
    
    print(f"✓ Loaded {num_batches} batches in {elapsed:.2f}s")
    print(f"✓ Throughput: {images_per_sec:.1f} images/sec")
    
    # Check class distribution (sample)
    print("\n6. Checking class distribution (first 1000 samples)...")
    class_counts = {}
    for i, (_, label) in enumerate(dm.train_dataset):
        if i >= 1000:
            break
        label_int = int(label)
        class_counts[label_int] = class_counts.get(label_int, 0) + 1
    
    print(f"✓ Found {len(class_counts)} unique classes in sample")
    print(f"✓ Sample class distribution: {list(class_counts.values())[:10]}...")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    
    return dm


def test_augmentations():
    """Test different augmentation settings."""
    print("\n" + "=" * 60)
    print("Testing Augmentation Variations")
    print("=" * 60)
    
    configs = [
        {"name": "No augmentation", "random_crop": False, "random_horizontal_flip": False},
        {"name": "Basic augmentation", "random_crop": True, "random_horizontal_flip": True},
        {"name": "With AutoAugment", "random_crop": True, "random_horizontal_flip": True, "auto_augment": "imagenet"},
    ]
    
    for config in configs:
        name = config.pop("name")
        print(f"\n{name}:")
        
        dm = ImageNetDataModule(
            data_root="/data2/imagenet",
            batch_size=32,
            num_workers=2,
            **config
        )
        dm.setup("fit")
        
        # Get a batch
        train_loader = dm.train_dataloader()
        images, _ = next(iter(train_loader))
        
        print(f"  ✓ Shape: {images.shape}")
        print(f"  ✓ Range: [{images.min():.3f}, {images.max():.3f}]")


if __name__ == "__main__":
    # Run basic tests
    dm = test_datamodule()
    
    # Test augmentations
    test_augmentations()
    
    print("\n🎉 All DataModule tests completed successfully!")
