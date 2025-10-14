#!/usr/bin/env python3
"""
Data integrity tests for ImageNet dataset.
Validates dataset structure, checks for corrupted images, and verifies class distribution.
"""
import os
from pathlib import Path
from collections import defaultdict
from PIL import Image
import argparse


def test_directory_structure(data_root: str):
    """Test that the dataset has the correct directory structure."""
    print("\n" + "=" * 70)
    print("Test 1: Directory Structure")
    print("=" * 70)
    
    data_path = Path(data_root)
    train_path = data_path / "train"
    val_path = data_path / "val"
    
    # Check main directories exist
    assert data_path.exists(), f"Data root does not exist: {data_root}"
    assert train_path.exists(), f"Train directory does not exist: {train_path}"
    assert val_path.exists(), f"Val directory does not exist: {val_path}"
    print(f"✓ Main directories exist")
    
    # Check number of classes
    train_classes = sorted([d.name for d in train_path.iterdir() if d.is_dir()])
    val_classes = sorted([d.name for d in val_path.iterdir() if d.is_dir()])
    
    print(f"✓ Train classes: {len(train_classes)}")
    print(f"✓ Val classes: {len(val_classes)}")
    
    assert len(train_classes) == 1000, f"Expected 1000 train classes, got {len(train_classes)}"
    assert len(val_classes) == 1000, f"Expected 1000 val classes, got {len(val_classes)}"
    print(f"✓ Both splits have 1000 classes")
    
    # Check class names match
    assert train_classes == val_classes, "Train and val class names don't match"
    print(f"✓ Train and val class names match")
    
    # Check for empty directories
    empty_train = [c for c in train_classes if not any((train_path / c).iterdir())]
    empty_val = [c for c in val_classes if not any((val_path / c).iterdir())]
    
    if empty_train:
        print(f"⚠ Warning: {len(empty_train)} empty train directories: {empty_train[:5]}...")
    else:
        print(f"✓ No empty train directories")
    
    if empty_val:
        print(f"⚠ Warning: {len(empty_val)} empty val directories: {empty_val[:5]}...")
    else:
        print(f"✓ No empty val directories")
    
    return train_classes, val_classes


def test_image_validity(data_root: str, sample_size: int = 1000):
    """Test that images can be loaded and are valid."""
    print("\n" + "=" * 70)
    print(f"Test 2: Image Validity (sampling {sample_size} images)")
    print("=" * 70)
    
    data_path = Path(data_root)
    train_path = data_path / "train"
    
    # Collect sample images
    all_images = []
    for class_dir in train_path.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*.JPEG"))
            all_images.extend(images[:min(10, len(images))])  # Sample 10 per class
            if len(all_images) >= sample_size:
                break
    
    all_images = all_images[:sample_size]
    print(f"Testing {len(all_images)} images...")
    
    corrupted = []
    valid = 0
    
    for img_path in all_images:
        try:
            with Image.open(img_path) as img:
                img.verify()  # Verify it's a valid image
            valid += 1
        except Exception as e:
            corrupted.append((str(img_path), str(e)))
    
    print(f"✓ Valid images: {valid}/{len(all_images)}")
    
    if corrupted:
        print(f"⚠ Corrupted images: {len(corrupted)}")
        print("First 5 corrupted images:")
        for path, error in corrupted[:5]:
            print(f"  - {path}: {error}")
    else:
        print(f"✓ No corrupted images found")
    
    corruption_rate = len(corrupted) / len(all_images) * 100
    print(f"✓ Corruption rate: {corruption_rate:.2f}%")
    
    assert corruption_rate < 1.0, f"Corruption rate too high: {corruption_rate:.2f}%"
    
    return corrupted


def test_class_distribution(data_root: str):
    """Test class distribution in train and val sets."""
    print("\n" + "=" * 70)
    print("Test 3: Class Distribution")
    print("=" * 70)
    
    data_path = Path(data_root)
    train_path = data_path / "train"
    val_path = data_path / "val"
    
    # Count images per class
    print("Counting images per class...")
    
    train_counts = {}
    val_counts = {}
    
    # Train set
    for class_dir in sorted(train_path.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*.JPEG")))
            train_counts[class_dir.name] = count
    
    # Val set
    for class_dir in sorted(val_path.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*.JPEG")))
            val_counts[class_dir.name] = count
    
    # Train statistics
    train_total = sum(train_counts.values())
    train_min = min(train_counts.values())
    train_max = max(train_counts.values())
    train_avg = train_total / len(train_counts)
    
    print(f"\nTrain Set:")
    print(f"  Total images: {train_total:,}")
    print(f"  Images per class: min={train_min}, max={train_max}, avg={train_avg:.1f}")
    
    # Val statistics
    val_total = sum(val_counts.values())
    val_min = min(val_counts.values())
    val_max = max(val_counts.values())
    val_avg = val_total / len(val_counts)
    
    print(f"\nValidation Set:")
    print(f"  Total images: {val_total:,}")
    print(f"  Images per class: min={val_min}, max={val_max}, avg={val_avg:.1f}")
    
    # Check validation set has exactly 50 per class
    non_50_classes = [c for c, count in val_counts.items() if count != 50]
    if non_50_classes:
        print(f"\n⚠ Warning: {len(non_50_classes)} classes don't have exactly 50 val images")
        print(f"  Examples: {non_50_classes[:5]}")
        for c in non_50_classes[:5]:
            print(f"    {c}: {val_counts[c]} images")
    else:
        print(f"\n✓ All validation classes have exactly 50 images")
    
    # Check for severely imbalanced classes in train
    imbalanced = [(c, count) for c, count in train_counts.items() 
                  if count < train_avg * 0.5 or count > train_avg * 2.0]
    
    if imbalanced:
        print(f"\n⚠ Warning: {len(imbalanced)} severely imbalanced train classes")
        print(f"  Examples (showing first 5):")
        for c, count in imbalanced[:5]:
            print(f"    {c}: {count} images (avg: {train_avg:.1f})")
    else:
        print(f"\n✓ No severely imbalanced train classes")
    
    return train_counts, val_counts


def test_file_naming(data_root: str, sample_size: int = 100):
    """Test that files follow expected naming conventions."""
    print("\n" + "=" * 70)
    print(f"Test 4: File Naming Conventions (sampling {sample_size} files)")
    print("=" * 70)
    
    data_path = Path(data_root)
    train_path = data_path / "train"
    
    # Sample files
    sample_files = []
    for class_dir in train_path.iterdir():
        if class_dir.is_dir():
            files = list(class_dir.glob("*.JPEG"))
            sample_files.extend(files[:5])
            if len(sample_files) >= sample_size:
                break
    
    sample_files = sample_files[:sample_size]
    
    # Check extensions
    extensions = defaultdict(int)
    for f in sample_files:
        extensions[f.suffix] += 1
    
    print(f"File extensions found:")
    for ext, count in extensions.items():
        print(f"  {ext}: {count} files")
    
    # Check for non-JPEG files
    non_jpeg = [f for f in sample_files if f.suffix.lower() not in ['.jpeg', '.jpg']]
    if non_jpeg:
        print(f"\n⚠ Warning: {len(non_jpeg)} non-JPEG files found")
        print(f"  Examples: {[f.name for f in non_jpeg[:5]]}")
    else:
        print(f"\n✓ All sampled files are JPEG")
    
    return extensions


def generate_report(data_root: str, output_file: str = "data_integrity_report.txt"):
    """Generate a comprehensive data integrity report."""
    print("\n" + "=" * 70)
    print("Generating Comprehensive Report")
    print("=" * 70)
    
    with open(output_file, "w") as f:
        f.write("ImageNet Data Integrity Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Data Root: {data_root}\n\n")
        
        # Run all tests and capture results
        try:
            train_classes, val_classes = test_directory_structure(data_root)
            f.write(f"✓ Directory structure: PASS\n")
            f.write(f"  Train classes: {len(train_classes)}\n")
            f.write(f"  Val classes: {len(val_classes)}\n\n")
        except AssertionError as e:
            f.write(f"✗ Directory structure: FAIL - {e}\n\n")
        
        try:
            corrupted = test_image_validity(data_root)
            f.write(f"✓ Image validity: PASS\n")
            f.write(f"  Corrupted images: {len(corrupted)}\n\n")
        except AssertionError as e:
            f.write(f"✗ Image validity: FAIL - {e}\n\n")
        
        try:
            train_counts, val_counts = test_class_distribution(data_root)
            f.write(f"✓ Class distribution: PASS\n")
            f.write(f"  Train total: {sum(train_counts.values()):,}\n")
            f.write(f"  Val total: {sum(val_counts.values()):,}\n\n")
        except Exception as e:
            f.write(f"✗ Class distribution: FAIL - {e}\n\n")
        
        try:
            extensions = test_file_naming(data_root)
            f.write(f"✓ File naming: PASS\n")
            f.write(f"  Extensions: {dict(extensions)}\n\n")
        except Exception as e:
            f.write(f"✗ File naming: FAIL - {e}\n\n")
    
    print(f"\n✓ Report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Test ImageNet data integrity")
    parser.add_argument("--data_root", type=str, default="/data2/imagenet",
                        help="Path to ImageNet dataset")
    parser.add_argument("--sample_size", type=int, default=1000,
                        help="Number of images to sample for validity test")
    parser.add_argument("--generate_report", action="store_true",
                        help="Generate a comprehensive report file")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("ImageNet Data Integrity Tests")
    print("=" * 70)
    print(f"Data root: {args.data_root}")
    print(f"Sample size: {args.sample_size}")
    
    try:
        # Run all tests
        test_directory_structure(args.data_root)
        test_image_validity(args.data_root, args.sample_size)
        test_class_distribution(args.data_root)
        test_file_naming(args.data_root)
        
        if args.generate_report:
            generate_report(args.data_root)
        
        print("\n" + "=" * 70)
        print("✅ All data integrity tests passed!")
        print("=" * 70)
        
    except AssertionError as e:
        print("\n" + "=" * 70)
        print(f"❌ Test failed: {e}")
        print("=" * 70)
        return 1
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ Unexpected error: {e}")
        print("=" * 70)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
