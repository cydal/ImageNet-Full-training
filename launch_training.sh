#!/bin/bash
# Launch full ImageNet training with ResNet Strikes Back recipe
# Optimized for 8x A100 40GB GPUs with 96 vCPUs

set -e

# Configuration
RUN_NAME="rsb_full_8xA100_$(date +%Y%m%d_%H%M%S)"
WANDB_PROJECT="imagenet-resnet50"
CONFIG="configs/resnet_strikes_back.yaml"

# Create logs directory
mkdir -p logs checkpoints

# Log file
LOG_FILE="logs/train_${RUN_NAME}.log"

echo "=========================================="
echo "Starting ImageNet Training"
echo "=========================================="
echo "Config: ${CONFIG}"
echo "Run name: ${RUN_NAME}"
echo "W&B project: ${WANDB_PROJECT}"
echo "Log file: ${LOG_FILE}"
echo ""
echo "Resource allocation:"
echo "  - GPUs: 8x A100 40GB"
echo "  - Batch size: 512 per GPU (4096 total)"
echo "  - Workers: 10 per GPU (80 total)"
echo "  - Learning rate: 8.0 (scaled)"
echo "  - Epochs: 600"
echo ""
echo "Starting training in background with nohup..."
echo "=========================================="

# Launch training with nohup
nohup python -u train.py \
  --config ${CONFIG} \
  --wandb_project ${WANDB_PROJECT} \
  --wandb_name ${RUN_NAME} \
  > ${LOG_FILE} 2>&1 &

# Save PID
PID=$!
echo ${PID} > logs/train.pid

echo ""
echo "✓ Training started!"
echo "  PID: ${PID}"
echo "  Log: ${LOG_FILE}"
echo ""
echo "Monitor progress:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "W&B dashboard:"
echo "  https://wandb.ai/<your-username>/${WANDB_PROJECT}/runs/${RUN_NAME}"
echo ""
echo "Check GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""
