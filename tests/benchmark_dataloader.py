#!/usr/bin/env python3
"""
Benchmark script for ImageNet DataLoader.
Tests different configurations to find optimal settings.
"""
import time
import argparse
import torch
from data.datamodule import ImageNetDataModule


def benchmark_config(
    data_root: str,
    batch_size: int,
    num_workers: int,
    persistent_workers: bool,
    pin_memory: bool,
    num_batches: int = 100
):
    """Benchmark a specific dataloader configuration."""
    
    # Create data module
    dm = ImageNetDataModule(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory
    )
    
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    
    # Warmup
    warmup_batches = min(10, num_batches // 10)
    for i, batch in enumerate(train_loader):
        if i >= warmup_batches:
            break
    
    # Benchmark
    start_time = time.time()
    total_images = 0
    
    for i, batch in enumerate(train_loader):
        if i >= num_batches:
            break
        images, labels = batch
        total_images += images.size(0)
    
    elapsed = time.time() - start_time
    throughput = total_images / elapsed
    
    return {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "persistent_workers": persistent_workers,
        "pin_memory": pin_memory,
        "total_images": total_images,
        "elapsed": elapsed,
        "throughput": throughput
    }


def benchmark_num_workers(data_root: str, batch_size: int = 256):
    """Benchmark different num_workers values."""
    print("\n" + "=" * 70)
    print(f"Benchmarking num_workers (batch_size={batch_size})")
    print("=" * 70)
    
    worker_counts = [0, 2, 4, 8, 12, 16]
    results = []
    
    for num_workers in worker_counts:
        print(f"\nTesting num_workers={num_workers}...")
        
        try:
            result = benchmark_config(
                data_root=data_root,
                batch_size=batch_size,
                num_workers=num_workers,
                persistent_workers=True if num_workers > 0 else False,
                pin_memory=True,
                num_batches=100
            )
            
            results.append(result)
            
            print(f"  Throughput: {result['throughput']:.1f} images/sec")
            print(f"  Time: {result['elapsed']:.2f}s for {result['total_images']} images")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    # Print summary
    print("\n" + "-" * 70)
    print("Summary:")
    print("-" * 70)
    print(f"{'num_workers':<15} {'Throughput (img/s)':<20} {'Time (s)':<15}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['num_workers']:<15} {r['throughput']:<20.1f} {r['elapsed']:<15.2f}")
    
    # Find best
    best = max(results, key=lambda x: x['throughput'])
    print("-" * 70)
    print(f"✓ Best: num_workers={best['num_workers']} with {best['throughput']:.1f} images/sec")
    
    return results


def benchmark_batch_size(data_root: str, num_workers: int = 8):
    """Benchmark different batch sizes."""
    print("\n" + "=" * 70)
    print(f"Benchmarking batch_size (num_workers={num_workers})")
    print("=" * 70)
    
    batch_sizes = [32, 64, 128, 256, 512]
    results = []
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch_size={batch_size}...")
        
        try:
            result = benchmark_config(
                data_root=data_root,
                batch_size=batch_size,
                num_workers=num_workers,
                persistent_workers=True,
                pin_memory=True,
                num_batches=100
            )
            
            results.append(result)
            
            print(f"  Throughput: {result['throughput']:.1f} images/sec")
            print(f"  Time: {result['elapsed']:.2f}s for {result['total_images']} images")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    # Print summary
    print("\n" + "-" * 70)
    print("Summary:")
    print("-" * 70)
    print(f"{'batch_size':<15} {'Throughput (img/s)':<20} {'Time (s)':<15}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['batch_size']:<15} {r['throughput']:<20.1f} {r['elapsed']:<15.2f}")
    
    print("-" * 70)
    
    return results


def benchmark_options(data_root: str):
    """Benchmark different dataloader options."""
    print("\n" + "=" * 70)
    print("Benchmarking DataLoader Options")
    print("=" * 70)
    
    configs = [
        {"name": "Baseline", "persistent_workers": False, "pin_memory": False},
        {"name": "Pin Memory", "persistent_workers": False, "pin_memory": True},
        {"name": "Persistent Workers", "persistent_workers": True, "pin_memory": False},
        {"name": "Both", "persistent_workers": True, "pin_memory": True},
    ]
    
    results = []
    
    for config in configs:
        name = config.pop("name")
        print(f"\nTesting: {name}")
        print(f"  Config: {config}")
        
        try:
            result = benchmark_config(
                data_root=data_root,
                batch_size=256,
                num_workers=8,
                num_batches=100,
                **config
            )
            
            result["name"] = name
            results.append(result)
            
            print(f"  Throughput: {result['throughput']:.1f} images/sec")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    # Print summary
    print("\n" + "-" * 70)
    print("Summary:")
    print("-" * 70)
    print(f"{'Configuration':<25} {'Throughput (img/s)':<20}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['name']:<25} {r['throughput']:<20.1f}")
    
    print("-" * 70)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark ImageNet DataLoader")
    parser.add_argument("--data_root", type=str, default="/data2/imagenet",
                        help="Path to ImageNet dataset")
    parser.add_argument("--test", type=str, default="all",
                        choices=["all", "workers", "batch_size", "options"],
                        help="Which benchmark to run")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for worker benchmark")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of workers for batch size benchmark")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("ImageNet DataLoader Benchmark")
    print("=" * 70)
    print(f"Data root: {args.data_root}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    try:
        if args.test in ["all", "workers"]:
            benchmark_num_workers(args.data_root, args.batch_size)
        
        if args.test in ["all", "batch_size"]:
            benchmark_batch_size(args.data_root, args.num_workers)
        
        if args.test in ["all", "options"]:
            benchmark_options(args.data_root)
        
        print("\n" + "=" * 70)
        print("✅ Benchmark completed!")
        print("=" * 70)
        
        print("\n📊 Recommendations:")
        print("  1. Use the num_workers value with highest throughput")
        print("  2. Enable both persistent_workers and pin_memory")
        print("  3. Adjust batch_size based on GPU memory")
        print("  4. Monitor GPU utilization during training")
        
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ Benchmark failed: {e}")
        print("=" * 70)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
