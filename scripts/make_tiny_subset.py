#!/usr/bin/env python3
"""
Create a tiny ImageNet subset for smoke testing.
Creates symlinks to a subset of classes from the full ImageNet dataset.

Usage:
    python scripts/make_tiny_subset.py --source /fsx/imagenet --target /fsx/imagenet-tiny --num_classes 10
"""
import argparse
import os
import shutil
from pathlib import Path
import random


def create_tiny_subset(source_dir: str, target_dir: str, num_classes: int = 10, num_train_images: int = 100, num_val_images: int = 50):
    """
    Create a tiny ImageNet subset by symlinking a subset of classes.
    
    Args:
        source_dir: Path to full ImageNet dataset
        target_dir: Path to create tiny subset
        num_classes: Number of classes to include
        num_train_images: Number of training images per class
        num_val_images: Number of validation images per class
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Check source exists
    if not source_path.exists():
        raise ValueError(f"Source directory does not exist: {source_dir}")
    
    # Create target directory
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of classes from training set
    train_source = source_path / "train"
    if not train_source.exists():
        raise ValueError(f"Training directory does not exist: {train_source}")
    
    all_classes = sorted([d.name for d in train_source.iterdir() if d.is_dir()])
    
    if len(all_classes) < num_classes:
        print(f"Warning: Only {len(all_classes)} classes available, using all of them")
        num_classes = len(all_classes)
    
    # Randomly select classes
    random.seed(42)
    selected_classes = random.sample(all_classes, num_classes)
    
    print(f"Selected {num_classes} classes: {selected_classes[:5]}...")
    
    # Create train subset
    print("\nCreating training subset...")
    train_target = target_path / "train"
    train_target.mkdir(parents=True, exist_ok=True)
    
    for class_name in selected_classes:
        class_source = train_source / class_name
        class_target = train_target / class_name
        
        # Get all images in this class
        images = sorted([f for f in class_source.iterdir() if f.is_file()])
        
        # Select subset
        selected_images = images[:num_train_images]
        
        # Create class directory
        class_target.mkdir(parents=True, exist_ok=True)
        
        # Create symlinks
        for img in selected_images:
            link_path = class_target / img.name
            if not link_path.exists():
                link_path.symlink_to(img)
        
        print(f"  {class_name}: {len(selected_images)} images")
    
    # Create val subset
    print("\nCreating validation subset...")
    val_source = source_path / "val"
    val_target = target_path / "val"
    val_target.mkdir(parents=True, exist_ok=True)
    
    for class_name in selected_classes:
        class_source = val_source / class_name
        class_target = val_target / class_name
        
        if not class_source.exists():
            print(f"  Warning: {class_name} not found in validation set, skipping")
            continue
        
        # Get all images in this class
        images = sorted([f for f in class_source.iterdir() if f.is_file()])
        
        # Select subset
        selected_images = images[:num_val_images]
        
        # Create class directory
        class_target.mkdir(parents=True, exist_ok=True)
        
        # Create symlinks
        for img in selected_images:
            link_path = class_target / img.name
            if not link_path.exists():
                link_path.symlink_to(img)
        
        print(f"  {class_name}: {len(selected_images)} images")
    
    print(f"\nTiny ImageNet subset created at: {target_path}")
    print(f"  Classes: {num_classes}")
    print(f"  Train images per class: ~{num_train_images}")
    print(f"  Val images per class: ~{num_val_images}")


def main():
    parser = argparse.ArgumentParser(description="Create tiny ImageNet subset for smoke testing")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to full ImageNet dataset")
    parser.add_argument("--target", type=str, required=True,
                        help="Path to create tiny subset")
    parser.add_argument("--num_classes", type=int, default=10,
                        help="Number of classes to include")
    parser.add_argument("--num_train_images", type=int, default=100,
                        help="Number of training images per class")
    parser.add_argument("--num_val_images", type=int, default=50,
                        help="Number of validation images per class")
    
    args = parser.parse_args()
    
    create_tiny_subset(
        args.source,
        args.target,
        args.num_classes,
        args.num_train_images,
        args.num_val_images
    )


if __name__ == "__main__":
    main()
