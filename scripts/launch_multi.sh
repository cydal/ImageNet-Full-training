#!/bin/bash
# Launch multi-node training with torchrun
# Reads host list from HOSTS file or environment variable

set -e

# Configuration
CONFIG="${CONFIG:-configs/base.yaml}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}"
MASTER_PORT="${MASTER_PORT:-29500}"
HOSTS_FILE="${HOSTS_FILE:-hosts.txt}"

echo "=========================================="
echo "Launching multi-node training"
echo "=========================================="

# Read hosts from file or environment
if [ -n "$MASTER_ADDR" ]; then
    echo "Using MASTER_ADDR from environment: $MASTER_ADDR"
elif [ -f "$HOSTS_FILE" ]; then
    echo "Reading hosts from: $HOSTS_FILE"
    MASTER_ADDR=$(head -n 1 "$HOSTS_FILE")
    NUM_NODES=$(wc -l < "$HOSTS_FILE")
    echo "Master node: $MASTER_ADDR"
    echo "Number of nodes: $NUM_NODES"
else
    echo "Error: MASTER_ADDR not set and $HOSTS_FILE not found"
    echo "Please set MASTER_ADDR or create $HOSTS_FILE with one hostname per line"
    exit 1
fi

# Get current node rank
CURRENT_HOST=$(hostname)
if [ -f "$HOSTS_FILE" ]; then
    NODE_RANK=$(grep -n "^$CURRENT_HOST$" "$HOSTS_FILE" | cut -d: -f1)
    NODE_RANK=$((NODE_RANK - 1))  # Convert to 0-indexed
else
    NODE_RANK="${NODE_RANK:-0}"
fi

echo "Config: $CONFIG"
echo "GPUs per node: $NUM_GPUS"
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Current Node: $CURRENT_HOST"
echo "Node Rank: $NODE_RANK"
echo "=========================================="

# Launch with torchrun
torchrun \
    --nnodes="$NUM_NODES" \
    --nproc_per_node="$NUM_GPUS" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    train.py \
    --config "$CONFIG" \
    --num_nodes="$NUM_NODES" \
    "$@"
