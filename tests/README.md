# Tests Directory

This directory contains all test files for the ImageNet training pipeline.

## Test Files

### Quick Tests
- **`quick_test.py`** - Fast pipeline verification (data → model → loss → gradients)
  - Runtime: ~30 seconds
  - Purpose: Verify end-to-end pipeline works
  - Run: `python tests/quick_test.py`

### Data Module Tests
- **`test_datamodule.py`** - Comprehensive data loading tests
  - Runtime: ~5-10 minutes
  - Purpose: Validate data loading, preprocessing, and throughput
  - Run: `python tests/test_datamodule.py`

- **`test_data_integrity.py`** - Data integrity and validation tests
  - Runtime: ~2-5 minutes
  - Purpose: Check dataset structure, corrupted images, class distribution
  - Run: `python tests/test_data_integrity.py`

### Model Tests
- **`test_model.py`** - Model architecture and forward pass tests
  - Runtime: ~1 minute
  - Purpose: Verify model structure, output shapes, parameter counts
  - Run: `python tests/test_model.py`

### Training Tests
- **`test_training_step.py`** - Training step and optimization tests
  - Runtime: ~2-3 minutes
  - Purpose: Verify loss computation, backward pass, optimizer step
  - Run: `python tests/test_training_step.py`

## Running Tests

### Run All Tests
```bash
# From project root
python -m pytest tests/ -v

# Or run individually
python tests/quick_test.py
python tests/test_datamodule.py
python tests/test_data_integrity.py
python tests/test_model.py
python tests/test_training_step.py
```

### Run Specific Test Categories
```bash
# Quick smoke test
make quick-test

# Data module tests
make test-data

# All tests (coming soon)
make test-all
```

## Test Coverage

### ✅ Implemented
- [x] Quick pipeline test
- [x] Data module basic tests
- [x] Data loading throughput benchmark

### 🔄 In Progress
- [ ] Data integrity tests
- [ ] Model architecture tests
- [ ] Training step tests

### ⏳ Planned
- [ ] Multi-GPU tests
- [ ] Distributed training tests
- [ ] Checkpoint save/load tests
- [ ] Augmentation tests
- [ ] Performance regression tests

## Test Data Module Plan

See `test_plan_datamodule.md` for detailed testing strategy.
