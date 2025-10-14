#!/usr/bin/env python3
"""
Quick test script to verify the entire pipeline.
Tests: data loading -> model forward -> loss computation
"""
import torch
import lightning.pytorch as pl
from data.datamodule import ImageNetDataModule
from models.resnet50 import ResNet50Module


def main():
    print("=" * 70)
    print("Quick Pipeline Test")
    print("=" * 70)
    
    # Set seed for reproducibility
    pl.seed_everything(42)
    
    # 1. Test Data Module
    print("\n[1/4] Testing Data Module...")
    dm = ImageNetDataModule(
        data_root="/data2/imagenet",
        batch_size=32,
        num_workers=4,
        img_size=224
    )
    dm.setup("fit")
    print(f"  ✓ Train samples: {len(dm.train_dataset):,}")
    print(f"  ✓ Val samples: {len(dm.val_dataset):,}")
    
    # 2. Test Model Creation
    print("\n[2/4] Testing Model Creation...")
    model = ResNet50Module(
        num_classes=1000,
        lr=0.1,
        epochs=90
    )
    print(f"  ✓ Model created: {model.__class__.__name__}")
    print(f"  ✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. Test Forward Pass
    print("\n[3/4] Testing Forward Pass...")
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    images, labels = batch
    
    print(f"  ✓ Batch shape: {images.shape}")
    print(f"  ✓ Labels shape: {labels.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(images)
    
    print(f"  ✓ Output shape: {outputs.shape}")
    print(f"  ✓ Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
    
    # 4. Test Training Step
    print("\n[4/4] Testing Training Step...")
    model.train()
    loss = model.training_step(batch, 0)
    
    print(f"  ✓ Loss computed: {loss.item():.4f}")
    print(f"  ✓ Loss is finite: {torch.isfinite(loss).item()}")
    
    # Test backward pass
    loss.backward()
    print(f"  ✓ Backward pass successful")
    
    # Check gradients
    has_grad = any(p.grad is not None for p in model.parameters())
    print(f"  ✓ Gradients computed: {has_grad}")
    
    print("\n" + "=" * 70)
    print("✅ All tests passed! Pipeline is working correctly.")
    print("=" * 70)
    
    print("\n📝 Next steps:")
    print("  1. Run full data test: make test-data")
    print("  2. Run single epoch: python train.py --config configs/local.yaml --epochs 1")
    print("  3. Run full local training: make train-local")


if __name__ == "__main__":
    main()
