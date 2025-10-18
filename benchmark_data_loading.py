#!/usr/bin/env python3
"""
Benchmark script to identify data loading bottlenecks.
Tests FSx throughput, JPEG decoding, and worker configurations.
"""
import time
import os
import sys
import torch
import numpy as np
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import multiprocessing as mp

print("=" * 80)
print("ImageNet Data Loading Benchmark")
print("=" * 80)
print()

# Configuration
DATA_ROOT = "/fsx/ns1/train"
BATCH_SIZE = 256
NUM_BATCHES_TO_TEST = 50  # Test 50 batches per config

print(f"Data root: {DATA_ROOT}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Testing {NUM_BATCHES_TO_TEST} batches per configuration")
print()

# Check data exists
if not os.path.exists(DATA_ROOT):
    print(f"ERROR: Data directory not found: {DATA_ROOT}")
    sys.exit(1)

print("=" * 80)
print("BENCHMARK 1: Raw FSx Read Speed")
print("=" * 80)
print("Testing raw file read performance from FSx...")

# Get some sample images
sample_dir = Path(DATA_ROOT)
sample_class = list(sample_dir.iterdir())[0]
sample_images = list(sample_class.glob("*.JPEG"))[:100]

if not sample_images:
    print("ERROR: No JPEG images found")
    sys.exit(1)

print(f"Testing with {len(sample_images)} sample images from {sample_class.name}")

# Test raw read speed
start = time.time()
total_bytes = 0
for img_path in sample_images:
    with open(img_path, 'rb') as f:
        data = f.read()
        total_bytes += len(data)
elapsed = time.time() - start

print(f"  Read {len(sample_images)} files in {elapsed:.2f}s")
print(f"  Total size: {total_bytes / 1024 / 1024:.1f} MB")
print(f"  Throughput: {total_bytes / 1024 / 1024 / elapsed:.1f} MB/s")
print(f"  Files/sec: {len(sample_images) / elapsed:.1f}")
print()

print("=" * 80)
print("BENCHMARK 2: JPEG Decoding Speed")
print("=" * 80)
print("Testing JPEG decoding performance (CPU-bound)...")

from PIL import Image

# Test JPEG decoding
start = time.time()
for img_path in sample_images:
    img = Image.open(img_path)
    img = img.convert('RGB')
    img.load()  # Force decode
elapsed = time.time() - start

print(f"  Decoded {len(sample_images)} JPEGs in {elapsed:.2f}s")
print(f"  Images/sec: {len(sample_images) / elapsed:.1f}")
print()

print("=" * 80)
print("BENCHMARK 3: Full Transform Pipeline")
print("=" * 80)
print("Testing complete augmentation pipeline...")

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

start = time.time()
for img_path in sample_images:
    img = Image.open(img_path).convert('RGB')
    tensor = transform(img)
elapsed = time.time() - start

print(f"  Processed {len(sample_images)} images in {elapsed:.2f}s")
print(f"  Images/sec: {len(sample_images) / elapsed:.1f}")
print()

print("=" * 80)
print("BENCHMARK 4: DataLoader with Different Worker Configs")
print("=" * 80)
print("Testing DataLoader throughput with various configurations...")
print()

# Create dataset
dataset = datasets.ImageFolder(DATA_ROOT, transform=transform)
print(f"Dataset size: {len(dataset):,} images")
print(f"Number of classes: {len(dataset.classes)}")
print()

# Test configurations
configs = [
    {"workers": 0, "prefetch": None, "name": "Single-threaded (baseline)"},
    {"workers": 4, "prefetch": 2, "name": "4 workers, prefetch=2"},
    {"workers": 8, "prefetch": 2, "name": "8 workers, prefetch=2"},
    {"workers": 12, "prefetch": 3, "name": "12 workers, prefetch=3 (current)"},
    {"workers": 16, "prefetch": 3, "name": "16 workers, prefetch=3"},
    {"workers": 12, "prefetch": 4, "name": "12 workers, prefetch=4"},
]

results = []

for config in configs:
    num_workers = config["workers"]
    prefetch = config["prefetch"]
    name = config["name"]
    
    print(f"\nTesting: {name}")
    print("-" * 60)
    
    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        if prefetch is not None:
            loader_kwargs["prefetch_factor"] = prefetch
    
    loader = DataLoader(dataset, **loader_kwargs)
    
    # Warmup
    print("  Warming up...")
    for i, (images, labels) in enumerate(loader):
        if i >= 3:
            break
    
    # Actual benchmark
    print("  Benchmarking...")
    start = time.time()
    batch_times = []
    
    for i, (images, labels) in enumerate(loader):
        batch_end = time.time()
        if i > 0:  # Skip first batch timing
            batch_times.append(batch_end - batch_start)
        batch_start = batch_end
        
        if i >= NUM_BATCHES_TO_TEST:
            break
    
    elapsed = time.time() - start
    
    # Calculate statistics
    batches = min(NUM_BATCHES_TO_TEST, len(loader))
    throughput = batches / elapsed
    images_per_sec = throughput * BATCH_SIZE
    
    if batch_times:
        avg_batch_time = np.mean(batch_times)
        std_batch_time = np.std(batch_times)
        min_batch_time = np.min(batch_times)
        max_batch_time = np.max(batch_times)
    else:
        avg_batch_time = std_batch_time = min_batch_time = max_batch_time = 0
    
    print(f"  Total time: {elapsed:.2f}s for {batches} batches")
    print(f"  Throughput: {throughput:.2f} batches/sec")
    print(f"  Images/sec: {images_per_sec:.0f}")
    print(f"  Avg batch time: {avg_batch_time:.3f}s ± {std_batch_time:.3f}s")
    print(f"  Min/Max batch time: {min_batch_time:.3f}s / {max_batch_time:.3f}s")
    
    results.append({
        "name": name,
        "workers": num_workers,
        "prefetch": prefetch,
        "throughput": throughput,
        "images_per_sec": images_per_sec,
        "avg_batch_time": avg_batch_time,
    })
    
    # Clean up
    del loader

print()
print("=" * 80)
print("BENCHMARK 5: GPU Transfer Speed")
print("=" * 80)
print("Testing CPU→GPU transfer speed...")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    
    # Create a sample batch
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4)
    images, labels = next(iter(loader))
    
    # Test transfer speed
    num_transfers = 20
    start = time.time()
    for _ in range(num_transfers):
        images_gpu = images.to(device, non_blocking=True)
        torch.cuda.synchronize()
    elapsed = time.time() - start
    
    batch_size_mb = images.element_size() * images.nelement() / 1024 / 1024
    print(f"  Batch size: {batch_size_mb:.1f} MB")
    print(f"  Transferred {num_transfers} batches in {elapsed:.2f}s")
    print(f"  Transfer speed: {batch_size_mb * num_transfers / elapsed:.1f} MB/s")
    print(f"  Batches/sec: {num_transfers / elapsed:.1f}")
else:
    print("  CUDA not available, skipping GPU transfer test")

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

print("DataLoader Performance Comparison:")
print("-" * 80)
print(f"{'Configuration':<40} {'Throughput':<15} {'Images/sec':<15} {'Batch Time':<15}")
print("-" * 80)

for result in results:
    print(f"{result['name']:<40} {result['throughput']:>6.2f} batch/s   {result['images_per_sec']:>8.0f} img/s   {result['avg_batch_time']:>6.3f}s")

print()
print("Recommendations:")
print("-" * 80)

# Find best configuration
best = max(results, key=lambda x: x['throughput'])
print(f"✓ Best configuration: {best['name']}")
print(f"  - Throughput: {best['throughput']:.2f} batches/sec ({best['images_per_sec']:.0f} images/sec)")
print(f"  - Average batch time: {best['avg_batch_time']:.3f}s")
print()

# Calculate expected training time
iterations_per_epoch = len(dataset) // BATCH_SIZE
time_per_epoch_min = (iterations_per_epoch / best['throughput']) / 60
time_100_epochs_days = (time_per_epoch_min * 100) / 60 / 24

print(f"Expected training time with best config:")
print(f"  - Iterations per epoch: {iterations_per_epoch:,}")
print(f"  - Time per epoch: {time_per_epoch_min:.1f} minutes")
print(f"  - Time for 100 epochs: {time_100_epochs_days:.1f} days (data loading only)")
print()

# Identify bottleneck
baseline = results[0]  # Single-threaded
best_multi = max(results[1:], key=lambda x: x['throughput'])
speedup = best_multi['throughput'] / baseline['throughput']

print(f"Speedup from parallelization: {speedup:.1f}x")
if speedup < 2:
    print("⚠️  WARNING: Low speedup suggests CPU or I/O bottleneck")
    print("   - JPEG decoding may be the bottleneck")
    print("   - Consider using NVIDIA DALI for GPU-accelerated data loading")
elif speedup < 4:
    print("⚠️  Moderate speedup - some parallelization benefit")
    print("   - May benefit from more workers or faster storage")
else:
    print("✓ Good speedup - parallelization is effective")

print()
print("=" * 80)
print("Benchmark complete!")
print("=" * 80)
