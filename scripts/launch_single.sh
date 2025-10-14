#!/bin/bash
# Launch single-node training with torchrun

set -e

# Configuration
CONFIG="${CONFIG:-configs/base.yaml}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}"
MASTER_PORT="${MASTER_PORT:-29500}"

echo "=========================================="
echo "Launching single-node training"
echo "=========================================="
echo "Config: $CONFIG"
echo "GPUs: $NUM_GPUS"
echo "Master Port: $MASTER_PORT"
echo "=========================================="

# Launch with torchrun
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="$MASTER_PORT" \
    train.py \
    --config "$CONFIG" \
    "$@"
