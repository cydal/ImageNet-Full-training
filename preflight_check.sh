#!/bin/bash
# Pre-flight check before starting full training

set -e

echo "=========================================="
echo "Pre-Flight Check for Full Training"
echo "=========================================="
echo ""

ERRORS=0
WARNINGS=0

# Check 1: FSx mounted
echo "[1/10] Checking FSx mount..."
if mountpoint -q /fsx 2>/dev/null; then
    echo "  ✓ FSx is mounted at /fsx"
else
    echo "  ✗ ERROR: FSx is not mounted"
    echo "    Run: ./scripts/mount_fsx.sh"
    ERRORS=$((ERRORS + 1))
fi

# Check 2: Data directories exist
echo "[2/10] Checking data directories..."
if [ -d "/fsx/ns1/train" ] && [ -d "/fsx/ns1/val" ]; then
    TRAIN_CLASSES=$(ls -1 /fsx/ns1/train | wc -l)
    VAL_CLASSES=$(ls -1 /fsx/ns1/val | wc -l)
    echo "  ✓ Train directory: $TRAIN_CLASSES classes"
    echo "  ✓ Val directory: $VAL_CLASSES classes"
    
    if [ "$TRAIN_CLASSES" -lt 1000 ] || [ "$VAL_CLASSES" -lt 1000 ]; then
        echo "  ⚠ WARNING: Expected 1000 classes, found train=$TRAIN_CLASSES, val=$VAL_CLASSES"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo "  ✗ ERROR: Data directories not found at /fsx/ns1/"
    ERRORS=$((ERRORS + 1))
fi

# Check 3: Python environment
echo "[3/10] Checking Python environment..."
if conda info --envs | grep -q "imagenet"; then
    echo "  ✓ Conda environment 'imagenet' exists"
    
    # Check if we're in the environment
    if [[ "$CONDA_DEFAULT_ENV" == "imagenet" ]]; then
        echo "  ✓ Currently in 'imagenet' environment"
    else
        echo "  ⚠ WARNING: Not in 'imagenet' environment"
        echo "    Run: conda activate imagenet"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo "  ✗ ERROR: Conda environment 'imagenet' not found"
    ERRORS=$((ERRORS + 1))
fi

# Check 4: Required packages
echo "[4/10] Checking required packages..."
REQUIRED_PACKAGES=("torch" "lightning" "torchvision" "wandb")
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if python -c "import $pkg" 2>/dev/null; then
        VERSION=$(python -c "import $pkg; print($pkg.__version__)" 2>/dev/null || echo "unknown")
        echo "  ✓ $pkg ($VERSION)"
    else
        echo "  ✗ ERROR: $pkg not installed"
        ERRORS=$((ERRORS + 1))
    fi
done

# Check 5: GPU availability
echo "[5/10] Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo "  ✓ Found $GPU_COUNT GPU(s): $GPU_NAME (${GPU_MEM}MB)"
    
    if [ "$GPU_MEM" -lt 16000 ]; then
        echo "  ⚠ WARNING: GPU has less than 16GB memory"
        echo "    Consider reducing batch size"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo "  ✗ ERROR: nvidia-smi not found (no GPU?)"
    ERRORS=$((ERRORS + 1))
fi

# Check 6: Disk space
echo "[6/10] Checking disk space..."
DISK_AVAIL=$(df -BG /home/ubuntu | tail -1 | awk '{print $4}' | sed 's/G//')
echo "  Available space: ${DISK_AVAIL}GB"
if [ "$DISK_AVAIL" -lt 100 ]; then
    echo "  ⚠ WARNING: Less than 100GB available"
    echo "    Checkpoints and logs will need space"
    WARNINGS=$((WARNINGS + 1))
else
    echo "  ✓ Sufficient disk space"
fi

# Check 7: W&B authentication
echo "[7/10] Checking W&B authentication..."
if [ -f "$HOME/.netrc" ] && grep -q "api.wandb.ai" "$HOME/.netrc"; then
    echo "  ✓ W&B API key configured"
else
    echo "  ⚠ WARNING: W&B not configured"
    echo "    Run: ./setup_wandb.sh"
    echo "    Or use: --no_wandb flag"
    WARNINGS=$((WARNINGS + 1))
fi

# Check 8: Config file
echo "[8/10] Checking config file..."
if [ -f "configs/single_gpu_full.yaml" ]; then
    echo "  ✓ Config file exists: configs/single_gpu_full.yaml"
else
    echo "  ✗ ERROR: Config file not found"
    ERRORS=$((ERRORS + 1))
fi

# Check 9: Checkpoint directory
echo "[9/10] Checking checkpoint directory..."
if [ -d "checkpoints" ]; then
    CKPT_COUNT=$(ls -1 checkpoints/*.ckpt 2>/dev/null | wc -l)
    if [ "$CKPT_COUNT" -gt 0 ]; then
        echo "  ⚠ WARNING: Found $CKPT_COUNT existing checkpoint(s)"
        echo "    Training will resume from last checkpoint"
        WARNINGS=$((WARNINGS + 1))
    else
        echo "  ✓ Checkpoint directory is empty (fresh start)"
    fi
else
    mkdir -p checkpoints
    echo "  ✓ Created checkpoint directory"
fi

# Check 10: Training script
echo "[10/10] Checking training script..."
if [ -f "train.py" ]; then
    echo "  ✓ Training script exists"
else
    echo "  ✗ ERROR: train.py not found"
    ERRORS=$((ERRORS + 1))
fi

# Summary
echo ""
echo "=========================================="
echo "Pre-Flight Check Summary"
echo "=========================================="
echo "Errors: $ERRORS"
echo "Warnings: $WARNINGS"
echo ""

if [ "$ERRORS" -gt 0 ]; then
    echo "❌ FAILED: Please fix errors before starting training"
    exit 1
elif [ "$WARNINGS" -gt 0 ]; then
    echo "⚠️  PASSED WITH WARNINGS: Review warnings above"
    echo ""
    echo "Ready to start training:"
    echo "  python train.py --config configs/single_gpu_full.yaml"
    exit 0
else
    echo "✅ ALL CHECKS PASSED"
    echo ""
    echo "Ready to start training:"
    echo "  python train.py --config configs/single_gpu_full.yaml"
    echo ""
    echo "With W&B logging:"
    echo "  python train.py \\"
    echo "    --config configs/single_gpu_full.yaml \\"
    echo "    --wandb_project imagenet-resnet50 \\"
    echo "    --wandb_name single-gpu-run-1"
    exit 0
fi
