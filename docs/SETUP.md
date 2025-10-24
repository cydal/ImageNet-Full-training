# Environment Setup Guide

## Current Status
⚠️ **Environment is NOT set up yet**

The codebase is ready, but dependencies need to be installed before running any tests or training.

## Quick Setup (5 minutes)

### Step 1: Install Dependencies
```bash
cd /home/ubuntu/imagenet
make install
```

This installs:
- `lightning>=2.0.0` (PyTorch Lightning)
- `torch>=2.0.0` (PyTorch)
- `torchvision>=0.15.0`
- `timm>=0.9.0` (Model library)
- `wandb>=0.15.0` (Logging)
- Other utilities (pyyaml, numpy, pillow, etc.)

### Step 2: Verify Installation
```bash
# Check Lightning
python -c "import lightning.pytorch as pl; print(f'✓ Lightning version: {pl.__version__}')"

# Check PyTorch
python -c "import torch; print(f'✓ PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')"

# Check Torchvision
python -c "import torchvision; print(f'✓ Torchvision version: {torchvision.__version__}')"

# Check data access
ls -la /data2/imagenet/train | head -5
ls -la /data2/imagenet/val | head -5
```

**Expected Output:**
```
✓ Lightning version: 2.x.x
✓ PyTorch version: 2.x.x
✓ CUDA available: True
✓ Torchvision version: 0.x.x
```

### Step 3: Run Quick Test
```bash
make quick-test
```

**Expected**: All tests pass (data loading → model → loss → gradients)

## Detailed Installation

### Option 1: Using Makefile (Recommended)
```bash
make install
```

### Option 2: Manual Installation
```bash
pip install -r requirements.txt
```

### Option 3: With Virtual Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Verify Setup

### 1. Check Python Version
```bash
python --version
# Should be Python 3.8 or higher
```

### 2. Check GPU
```bash
nvidia-smi
# Should show your GPU(s)
```

### 3. Check Data
```bash
# Check data exists
ls /data2/imagenet

# Count classes
ls /data2/imagenet/train | wc -l  # Should be 1000
ls /data2/imagenet/val | wc -l    # Should be 1000

# Check sample images
ls /data2/imagenet/train/n01440764 | head -5
```

## Common Issues

### Issue: "No module named 'lightning'"
**Solution:**
```bash
pip install lightning>=2.0.0
```

### Issue: "No module named 'torch'"
**Solution:**
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Issue: "FileNotFoundError: /data2/imagenet"
**Solution:**
```bash
# Check if mounted
df -h /data2

# Mount if needed
sudo mount /dev/xvdf /data2

# Verify
ls /data2/imagenet
```

### Issue: Import error with pytorch_lightning
**Solution:**
All code has been updated to use `lightning.pytorch` instead of `pytorch_lightning`.
If you see this error, make sure you're using the latest code.

## Package Versions

### Minimum Requirements
- Python >= 3.8
- PyTorch >= 2.0.0
- Lightning >= 2.0.0
- CUDA >= 11.8 (for GPU training)

### Tested Versions
- Python 3.10
- PyTorch 2.1.0
- Lightning 2.1.0
- CUDA 12.1

## Next Steps

After successful setup:

1. ✅ Run quick test: `make quick-test`
2. ✅ Run data tests: `make test-data`
3. ✅ Run integrity tests: `make test-integrity`
4. ✅ Benchmark dataloader: `make benchmark-data`
5. ✅ Single epoch training: `python train.py --config configs/local.yaml --epochs 1`
6. ✅ Full training: `make train-local`

## Installation Log

Keep track of your setup:

```bash
# Date: ___________
# Python version: ___________
# PyTorch version: ___________
# Lightning version: ___________
# CUDA version: ___________
# GPU: ___________

# Tests passed:
# [ ] make quick-test
# [ ] make test-data
# [ ] make test-integrity
# [ ] make benchmark-data

# Issues encountered:
# 
# 
# 

# Solutions applied:
# 
# 
# 
```

## Support

If you encounter issues:
1. Check this guide for common issues
2. Review `docs/02_local_development.md` for troubleshooting
3. Check test logs in the terminal output
4. Verify data path and permissions
