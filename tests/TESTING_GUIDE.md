# Testing Guide

## Overview
This guide explains how to test the ImageNet data module and training pipeline.

## Test Structure

```
tests/
├── __init__.py
├── README.md                    # Test overview
├── TESTING_GUIDE.md            # This file
├── test_plan_datamodule.md     # Detailed test plan
├── quick_test.py               # Quick pipeline verification
├── test_datamodule.py          # Data module tests
├── test_data_integrity.py      # Data integrity tests
└── benchmark_dataloader.py     # Performance benchmarks
```

## Quick Start

### 1. Quick Pipeline Test (30 seconds)
Verifies the entire pipeline works end-to-end.

```bash
make quick-test
```

**What it tests:**
- ✓ Data loading from `/data2/imagenet`
- ✓ Model creation (ResNet50)
- ✓ Forward pass
- ✓ Loss computation
- ✓ Backward pass and gradients

**Expected output:**
```
[1/4] Testing Data Module...
  ✓ Train samples: 1,281,167
  ✓ Val samples: 50,000

[2/4] Testing Model Creation...
  ✓ Model created: ResNet50Module
  ✓ Parameters: 25,557,032

[3/4] Testing Forward Pass...
  ✓ Batch shape: torch.Size([32, 3, 224, 224])
  ✓ Output shape: torch.Size([32, 1000])

[4/4] Testing Training Step...
  ✓ Loss computed: 6.9077
  ✓ Backward pass successful

✅ All tests passed!
```

### 2. Data Module Tests (5-10 minutes)
Comprehensive data loading tests.

```bash
make test-data
```

**What it tests:**
- Dataset sizes and structure
- Batch loading and shapes
- Data preprocessing
- Augmentation pipeline
- Loading throughput (50 batches)
- Class distribution sampling

**Expected output:**
- Train: ~1,281,167 images
- Val: 50,000 images
- Classes: 1000
- Throughput: 400-2000 images/sec (depends on CPU/GPU)

**Actual results (Oct 14, 2025):**
- ✅ Train: 1,281,167 images
- ✅ Val: 50,000 images
- ✅ Classes: 1000
- ✅ Throughput: 445 images/sec (CPU-only, will improve with GPU)

### 3. Data Integrity Tests (2-5 minutes)
Validates dataset quality and structure.

```bash
make test-integrity
```

**What it tests:**
- Directory structure (1000 classes in train/val)
- Image validity (checks for corrupted files)
- Class distribution (images per class)
- File naming conventions

**Expected output:**
- ✓ 1000 classes in both train and val
- ✓ No corrupted images (or < 0.1%)
- ✓ Val set: 50 images per class
- ✓ All files are JPEG format

**Generates:** `data_integrity_report.txt`

### 4. Dataloader Benchmark (10-15 minutes)
Performance benchmarking for optimization.

```bash
make benchmark-data
```

**What it tests:**
- Different `num_workers` values [0, 2, 4, 8, 12, 16]
- Different `batch_size` values [32, 64, 128, 256, 512]
- DataLoader options (persistent_workers, pin_memory)

**Expected output:**
- Throughput for each configuration
- Optimal `num_workers` recommendation
- Performance comparison table

### 5. Run All Tests (15-20 minutes)
Runs all tests in sequence.

```bash
make test-all
```

## Test Commands Reference

```bash
# Quick tests
make quick-test          # 30 seconds - pipeline verification
make test-data           # 5-10 min - data module tests
make test-integrity      # 2-5 min - data validation
make benchmark-data      # 10-15 min - performance tuning

# All tests
make test-all            # 15-20 min - everything

# Individual test files
python tests/quick_test.py
python tests/test_datamodule.py
python tests/test_data_integrity.py
python tests/benchmark_dataloader.py
```

## Interpreting Results

### Quick Test
**✅ Pass**: Pipeline is working, ready for training  
**❌ Fail**: Check error message, likely data path or dependency issue

### Data Module Tests
**✅ Pass**: Data loading is correct  
**⚠️ Low throughput (<500 img/s)**: Run benchmark to optimize  
**❌ Fail**: Check data path and structure

### Data Integrity Tests
**✅ Pass**: Dataset is valid  
**⚠️ Corrupted images found**: Review report, may need to skip/replace  
**⚠️ Class imbalance**: Expected for ImageNet, not critical  
**❌ Fail**: Dataset structure issue, check data

### Benchmark
**Good**: >1000 images/sec per GPU  
**Acceptable**: 500-1000 images/sec  
**Poor**: <500 images/sec - optimize num_workers or check storage

## Troubleshooting

### Test fails with "FileNotFoundError"
```bash
# Check if data exists
ls /data2/imagenet

# Check if mounted
df -h /data2

# Mount if needed
sudo mount /dev/xvdf /data2
```

### Test fails with "No module named..."
```bash
# Install dependencies
make install
```

### Low throughput in benchmark
**Try:**
1. Increase `num_workers` (8, 12, 16)
2. Enable `persistent_workers=True`
3. Enable `pin_memory=True`
4. Check disk I/O: `iostat -x 1`
5. Consider faster storage (SSD, FSx)

### Corrupted images found
**Options:**
1. Skip corrupted images (modify DataLoader)
2. Replace with valid images
3. If < 0.1%, ignore (minimal impact)

### Memory issues during testing
```bash
# Reduce batch size
python tests/test_datamodule.py --batch_size 32

# Reduce num_workers
python tests/benchmark_dataloader.py --num_workers 4
```

## Test Development Workflow

### Before Training
1. ✅ Run `make quick-test` - verify pipeline
2. ✅ Run `make test-integrity` - validate data
3. ✅ Run `make benchmark-data` - optimize settings
4. ✅ Update config with optimal settings

### After Code Changes
1. Run `make quick-test` - verify no breakage
2. If data module changed: `make test-data`
3. If model changed: test model separately

### Before Production
1. Run `make test-all` - full validation
2. Review all reports
3. Document any issues/workarounds
4. Verify on production hardware

## Performance Targets

### Data Loading
| Hardware | Target Throughput |
|----------|-------------------|
| HDD | 300-500 images/sec |
| SSD | 800-1500 images/sec |
| NVMe | 1500-3000 images/sec |
| FSx | 2000-5000 images/sec |

### Test Execution Time
| Test | Expected Time |
|------|---------------|
| quick-test | 30 seconds |
| test-data | 5-10 minutes |
| test-integrity | 2-5 minutes |
| benchmark-data | 10-15 minutes |
| test-all | 15-20 minutes |

## Next Steps

After all tests pass:
1. ✅ Verify optimal `num_workers` from benchmark
2. ✅ Update `configs/local.yaml` with optimal settings
3. ✅ Run single epoch training: `python train.py --config configs/local.yaml --epochs 1`
4. ✅ Monitor GPU utilization during training
5. ✅ Proceed to full training: `make train-local`

## Additional Resources

- **Test Plan**: `test_plan_datamodule.md` - Detailed testing strategy
- **Test README**: `README.md` - Test file descriptions
- **Development Guide**: `../docs/02_local_development.md` - Development workflow
- **Getting Started**: `../docs/03_getting_started.md` - Quick start guide
