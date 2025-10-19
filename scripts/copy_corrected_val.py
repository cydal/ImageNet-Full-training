#!/usr/bin/env python3
"""
Copy corrected validation data from S3 to local NVMe.
This script copies validation data for the same 100 classes already in training.
"""
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm


def get_train_classes(train_dir):
    """Get list of classes from training directory."""
    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    return classes


def copy_val_class(source_class_dir, dest_class_dir, max_samples=None):
    """
    Copy validation images for a single class.
    
    Args:
        source_class_dir: Source directory for this class
        dest_class_dir: Destination directory for this class
        max_samples: Maximum number of samples to copy (None = all)
    
    Returns:
        Tuple of (num_files_copied, total_bytes_copied)
    """
    # Get all images
    images = sorted(list(source_class_dir.glob("*.JPEG")))
    
    if max_samples is not None:
        images = images[:max_samples]
    
    # Create destination directory
    dest_class_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy files
    total_bytes = 0
    for img in images:
        dest = dest_class_dir / img.name
        shutil.copy2(img, dest)
        total_bytes += img.stat().st_size
    
    return len(images), total_bytes


def main():
    parser = argparse.ArgumentParser(
        description="Copy corrected validation data from S3 to local NVMe"
    )
    parser.add_argument("--source", type=str, default="/mnt/s3-imagenet/imagenet",
                        help="Source directory (S3 mount)")
    parser.add_argument("--dest", type=str, default="/mnt/nvme_data/imagenet_subset",
                        help="Destination directory (local NVMe)")
    parser.add_argument("--max_samples_val", type=int, default=10,
                        help="Max samples per class for validation")
    
    args = parser.parse_args()
    
    source_val = Path(args.source) / "val"
    dest_train = Path(args.dest) / "train"
    dest_val = Path(args.dest) / "val"
    
    print("="*70)
    print("Copy Corrected Validation Data")
    print("="*70)
    print(f"Source: {source_val}")
    print(f"Destination: {dest_val}")
    print(f"Val samples per class: {args.max_samples_val}")
    print("="*70)
    print()
    
    # Check if destination train exists
    if not dest_train.exists():
        print(f"ERROR: Training directory not found: {dest_train}")
        print("Please ensure training data exists first.")
        return
    
    # Check if destination val already exists
    if dest_val.exists():
        response = input(f"Destination {dest_val} exists. Delete and recreate? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return
        shutil.rmtree(dest_val)
        print(f"Deleted {dest_val}")
        print()
    
    # Get classes from training directory
    print("Reading classes from training directory...")
    classes = get_train_classes(dest_train)
    print(f"Found {len(classes)} classes in training set")
    print(f"First 10 classes: {classes[:10]}")
    print()
    
    # Verify all classes exist in source
    missing_classes = []
    for cls in classes:
        if not (source_val / cls).exists():
            missing_classes.append(cls)
    
    if missing_classes:
        print(f"WARNING: {len(missing_classes)} classes missing in source validation:")
        for cls in missing_classes[:10]:
            print(f"  - {cls}")
        if len(missing_classes) > 10:
            print(f"  ... and {len(missing_classes) - 10} more")
        print()
        response = input("Continue anyway? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return
    
    # Copy validation data
    print("Copying validation data...")
    total_files = 0
    total_bytes = 0
    
    for class_name in tqdm(classes, desc="Classes"):
        source_class = source_val / class_name
        dest_class = dest_val / class_name
        
        if not source_class.exists():
            print(f"\nSkipping {class_name} (not found in source)")
            continue
        
        n_files, n_bytes = copy_val_class(
            source_class, dest_class, args.max_samples_val
        )
        
        total_files += n_files
        total_bytes += n_bytes
    
    # Summary
    total_gb = total_bytes / (1024**3)
    
    print()
    print("="*70)
    print("Copy Complete!")
    print("="*70)
    print(f"Total files copied: {total_files:,}")
    print(f"Total size: {total_gb:.2f} GB")
    print(f"Classes: {len(classes)}")
    print(f"Avg samples per class: {total_files / len(classes):.1f}")
    print()
    print(f"Validation data location: {dest_val}")
    print()
    print("IMPORTANT: Verify the validation data is correctly labeled!")
    print("Run a quick test:")
    print(f"  python train.py --config configs/test_trainonly_val.yaml --no_wandb")
    print("="*70)


if __name__ == "__main__":
    main()
