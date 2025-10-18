#!/bin/bash
# Setup Weights & Biases for experiment tracking

set -e

echo "=========================================="
echo "Weights & Biases Setup"
echo "=========================================="
echo ""

# Check if wandb is installed
if ! python -c "import wandb" 2>/dev/null; then
    echo "Installing wandb..."
    pip install wandb
else
    echo "✓ wandb is already installed"
fi

echo ""
echo "Please enter your W&B API key:"
echo "(Find it at: https://wandb.ai/authorize)"
echo ""
read -sp "API Key: " WANDB_API_KEY
echo ""

# Login to wandb
echo "$WANDB_API_KEY" | wandb login

echo ""
echo "=========================================="
echo "✓ W&B Setup Complete!"
echo "=========================================="
echo ""
echo "Your API key has been saved to: ~/.netrc"
echo ""
echo "To start training with W&B logging:"
echo "  python train.py --config configs/single_gpu_full.yaml"
echo ""
echo "To disable W&B logging:"
echo "  python train.py --config configs/single_gpu_full.yaml --no_wandb"
echo ""
echo "To set custom project/run name:"
echo "  python train.py \\"
echo "    --config configs/single_gpu_full.yaml \\"
echo "    --wandb_project my-imagenet-project \\"
echo "    --wandb_name resnet50-experiment-1"
echo ""
