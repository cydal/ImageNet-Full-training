#!/usr/bin/env python3
"""
Copy ImageNet subset to local NVMe disk.

This script:
1. Identifies which classes are in the subset (based on subset_seed)
2. Copies only those classes from S3 to local disk
3. Shows progress and estimates time
"""
import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm
import argparse


def get_subset_classes(source_dir, max_classes, subset_seed):
    """
    Determine which classes are selected by the subsetting logic.
    This matches the logic in datamodule.py
    """
    # Get all class directories
    class_dirs = sorted([d for d in Path(source_dir).iterdir() if d.is_dir()])
    all_classes = [d.name for d in class_dirs]
    
    # Apply same random selection as datamodule
    random.seed(subset_seed)
    selected_classes = random.sample(all_classes, min(max_classes, len(all_classes)))
    
    return sorted(selected_classes)


def estimate_size(source_dir, classes, max_samples_per_class=None):
    """Estimate total size to copy."""
    total_size = 0
    total_files = 0
    
    print("Estimating size...")
    for class_name in tqdm(classes[:5], desc="Sampling classes"):  # Sample first 5 for estimate
        class_dir = Path(source_dir) / class_name
        if not class_dir.exists():
            continue
        
        files = list(class_dir.glob("*.JPEG"))
        if max_samples_per_class:
            files = files[:max_samples_per_class]
        
        for f in files:
            try:
                total_size += f.stat().st_size
                total_files += 1
            except:
                pass
    
    # Extrapolate to all classes
    if total_files > 0:
        avg_size_per_class = total_size / min(5, len(classes))
        estimated_total = avg_size_per_class * len(classes)
        avg_files_per_class = total_files / min(5, len(classes))
        estimated_files = int(avg_files_per_class * len(classes))
    else:
        estimated_total = 0
        estimated_files = 0
    
    return estimated_total, estimated_files


def copy_subset(source_dir, dest_dir, classes, max_samples_per_class=None):
    """Copy selected classes to destination."""
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    total_copied = 0
    total_size = 0
    
    for class_name in tqdm(classes, desc="Copying classes"):
        source_class = Path(source_dir) / class_name
        dest_class = dest_path / class_name
        
        if not source_class.exists():
            print(f"Warning: {source_class} not found, skipping")
            continue
        
        dest_class.mkdir(parents=True, exist_ok=True)
        
        # Get files
        files = sorted(source_class.glob("*.JPEG"))
        
        # Limit samples if requested
        if max_samples_per_class:
            random.seed(42 + hash(class_name))  # Deterministic per class
            files = random.sample(files, min(max_samples_per_class, len(files)))
        
        # Copy files
        for src_file in files:
            dest_file = dest_class / src_file.name
            if not dest_file.exists():
                shutil.copy2(src_file, dest_file)
                total_copied += 1
                total_size += src_file.stat().st_size
    
    return total_copied, total_size


def main():
    parser = argparse.ArgumentParser(description="Copy ImageNet subset to local disk")
    parser.add_argument("--source", type=str, default="/mnt/s3-imagenet/imagenet",
                        help="Source directory (S3 mount)")
    parser.add_argument("--dest", type=str, default="/data/imagenet_subset",
                        help="Destination directory (local NVMe)")
    parser.add_argument("--max_classes", type=int, default=100,
                        help="Number of classes to copy")
    parser.add_argument("--max_samples_train", type=int, default=500,
                        help="Max samples per class for training")
    parser.add_argument("--max_samples_val", type=int, default=10,
                        help="Max samples per class for validation")
    parser.add_argument("--subset_seed", type=int, default=42,
                        help="Random seed for class selection (must match config)")
    
    args = parser.parse_args()
    
    source_train = Path(args.source) / "train"
    source_val = Path(args.source) / "val"
    dest_train = Path(args.dest) / "train"
    dest_val = Path(args.dest) / "val"
    
    print("="*70)
    print("ImageNet Subset Copy to Local Disk")
    print("="*70)
    print(f"Source: {args.source}")
    print(f"Destination: {args.dest}")
    print(f"Classes: {args.max_classes}")
    print(f"Train samples per class: {args.max_samples_train}")
    print(f"Val samples per class: {args.max_samples_val}")
    print(f"Subset seed: {args.subset_seed}")
    print("="*70)
    print()
    
    # Get selected classes
    print("Identifying classes to copy...")
    classes = get_subset_classes(source_train, args.max_classes, args.subset_seed)
    print(f"Selected {len(classes)} classes")
    print(f"First 10 classes: {classes[:10]}")
    print()
    
    # Estimate size
    train_size, train_files = estimate_size(source_train, classes, args.max_samples_train)
    val_size, val_files = estimate_size(source_val, classes, args.max_samples_val)
    
    total_size_gb = (train_size + val_size) / (1024**3)
    total_files = train_files + val_files
    
    print(f"Estimated size: {total_size_gb:.2f} GB")
    print(f"Estimated files: {total_files:,}")
    print()
    
    # Check destination space
    dest_parent = Path(args.dest).parent
    if not dest_parent.exists():
        dest_parent = Path(args.dest).parts[0] if not args.dest.startswith('/') else '/' + Path(args.dest).parts[1]
    dest_stat = shutil.disk_usage(str(dest_parent) if dest_parent else '/')
    available_gb = dest_stat.free / (1024**3)
    print(f"Available space on destination: {available_gb:.2f} GB")
    
    if total_size_gb > available_gb * 0.9:
        print("ERROR: Not enough space on destination!")
        return
    
    print()
    response = input("Proceed with copy? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted.")
        return
    
    print()
    print("Copying training data...")
    train_copied, train_bytes = copy_subset(
        source_train, dest_train, classes, args.max_samples_train
    )
    
    print()
    print("Copying validation data...")
    val_copied, val_bytes = copy_subset(
        source_val, dest_val, classes, args.max_samples_val
    )
    
    total_copied = train_copied + val_copied
    total_bytes = train_bytes + val_bytes
    total_gb = total_bytes / (1024**3)
    
    print()
    print("="*70)
    print("Copy Complete!")
    print("="*70)
    print(f"Total files copied: {total_copied:,}")
    print(f"Total size: {total_gb:.2f} GB")
    print(f"Train: {train_copied:,} files ({train_bytes/(1024**3):.2f} GB)")
    print(f"Val: {val_copied:,} files ({val_bytes/(1024**3):.2f} GB)")
    print()
    print(f"Data location: {args.dest}")
    print()
    print("Update your config with:")
    print(f"  data_root: {args.dest}")
    print("="*70)


if __name__ == "__main__":
    main()
