#!/usr/bin/env python3
"""
Copy ImageNet subset using ONLY training images.
Split training images into train (450/class) and val (50/class).
This tests if the validation data is the problem.
"""
import argparse
import random
import shutil
from pathlib import Path
from tqdm import tqdm


def get_subset_classes(source_train, max_classes, seed):
    """Select a random subset of classes."""
    random.seed(seed)
    all_classes = sorted([d.name for d in source_train.iterdir() if d.is_dir()])
    selected = random.sample(all_classes, min(max_classes, len(all_classes)))
    return sorted(selected)


def copy_split_class(source_class_dir, dest_train_dir, dest_val_dir, 
                     n_train, n_val, seed):
    """
    Copy images from a single class, splitting into train and val.
    
    Args:
        source_class_dir: Source directory for this class (from train/)
        dest_train_dir: Destination train directory for this class
        dest_val_dir: Destination val directory for this class
        n_train: Number of images for training
        n_val: Number of images for validation
        seed: Random seed for reproducibility
    """
    # Get all images
    images = sorted(list(source_class_dir.glob("*.JPEG")))
    
    if len(images) < n_train + n_val:
        print(f"  Warning: {source_class_dir.name} has only {len(images)} images, "
              f"need {n_train + n_val}")
        n_train = min(n_train, len(images))
        n_val = min(n_val, len(images) - n_train)
    
    # Shuffle and split
    random.seed(seed)
    random.shuffle(images)
    train_images = images[:n_train]
    val_images = images[n_train:n_train + n_val]
    
    # Create destination directories
    dest_train_dir.mkdir(parents=True, exist_ok=True)
    dest_val_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy files
    train_bytes = 0
    val_bytes = 0
    
    for img in train_images:
        dest = dest_train_dir / img.name
        shutil.copy2(img, dest)
        train_bytes += img.stat().st_size
    
    for img in val_images:
        dest = dest_val_dir / img.name
        shutil.copy2(img, dest)
        val_bytes += img.stat().st_size
    
    return len(train_images), train_bytes, len(val_images), val_bytes


def main():
    parser = argparse.ArgumentParser(
        description="Copy ImageNet subset using ONLY train images (split into train/val)"
    )
    parser.add_argument("--source", type=str, default="/mnt/s3-imagenet/imagenet",
                        help="Source directory (S3 mount)")
    parser.add_argument("--dest", type=str, default="/mnt/nvme_data/imagenet_subset_trainonly",
                        help="Destination directory (local NVMe)")
    parser.add_argument("--max_classes", type=int, default=100,
                        help="Number of classes to copy")
    parser.add_argument("--train_per_class", type=int, default=450,
                        help="Images per class for training")
    parser.add_argument("--val_per_class", type=int, default=50,
                        help="Images per class for validation (from train set)")
    parser.add_argument("--subset_seed", type=int, default=42,
                        help="Random seed for class selection")
    
    args = parser.parse_args()
    
    source_train = Path(args.source) / "train"
    dest_train = Path(args.dest) / "train"
    dest_val = Path(args.dest) / "val"
    
    print("="*70)
    print("ImageNet Subset Copy (Train-Only Split)")
    print("="*70)
    print(f"Source: {args.source}/train")
    print(f"Destination: {args.dest}")
    print(f"Classes: {args.max_classes}")
    print(f"Train samples per class: {args.train_per_class}")
    print(f"Val samples per class: {args.val_per_class} (FROM TRAIN SET)")
    print(f"Subset seed: {args.subset_seed}")
    print()
    print("NOTE: Validation images come from training set to test if")
    print("      the original validation data is mislabeled.")
    print("="*70)
    print()
    
    # Check if destination exists
    if Path(args.dest).exists():
        response = input(f"Destination {args.dest} exists. Delete and recreate? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return
        shutil.rmtree(args.dest)
        print(f"Deleted {args.dest}")
        print()
    
    # Get selected classes
    print("Identifying classes to copy...")
    classes = get_subset_classes(source_train, args.max_classes, args.subset_seed)
    print(f"Selected {len(classes)} classes")
    print(f"First 10 classes: {classes[:10]}")
    print()
    
    # Estimate total
    total_images = len(classes) * (args.train_per_class + args.val_per_class)
    print(f"Will copy approximately {total_images:,} images")
    print()
    
    # Copy classes
    print("Copying classes...")
    total_train_files = 0
    total_train_bytes = 0
    total_val_files = 0
    total_val_bytes = 0
    
    for class_name in tqdm(classes, desc="Classes"):
        source_class = source_train / class_name
        dest_train_class = dest_train / class_name
        dest_val_class = dest_val / class_name
        
        n_train, train_bytes, n_val, val_bytes = copy_split_class(
            source_class, dest_train_class, dest_val_class,
            args.train_per_class, args.val_per_class,
            args.subset_seed
        )
        
        total_train_files += n_train
        total_train_bytes += train_bytes
        total_val_files += n_val
        total_val_bytes += val_bytes
    
    # Summary
    total_files = total_train_files + total_val_files
    total_bytes = total_train_bytes + total_val_bytes
    total_gb = total_bytes / (1024**3)
    
    print()
    print("="*70)
    print("Copy Complete!")
    print("="*70)
    print(f"Total files copied: {total_files:,}")
    print(f"Total size: {total_gb:.2f} GB")
    print(f"Train: {total_train_files:,} files ({total_train_bytes/(1024**3):.2f} GB)")
    print(f"Val: {total_val_files:,} files ({total_val_bytes/(1024**3):.2f} GB)")
    print()
    print(f"Data location: {args.dest}")
    print()
    print("IMPORTANT: Val images are from TRAIN set (correctly labeled)")
    print("Update your config with:")
    print(f"  data_root: {args.dest}")
    print("="*70)


if __name__ == "__main__":
    main()
