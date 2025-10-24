#!/bin/bash

cd /home/ubuntu/ImageNet-Full-training

RUN_NAME="rsb_a2_epoch124_lr2e-4_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/train_${RUN_NAME}.log"
CHECKPOINT="checkpoints/epoch124_lr2e-4_cosine.ckpt"

echo "=========================================="
echo "FRESH START FROM EPOCH 124"
echo "=========================================="
echo "Run name: ${RUN_NAME}"
echo "Log file: ${LOG_FILE}"
echo "Checkpoint: ${CHECKPOINT}"
echo ""
echo "CONFIGURATION (DATA-DRIVEN):"
echo "  - Starting epoch: 124"
echo "  - Starting accuracy: 67.5%"
echo "  - Initial LR: 2e-4 (0.0002)"
echo "  - LR Schedule: Cosine decay"
echo "  - Decay period: 476 epochs (124 → 600)"
echo "  - Minimum LR: 1e-5 (0.00001)"
echo "  - Found via LR range test at epoch 124"
echo ""
echo "LR SCHEDULE:"
echo "  - Epoch 124: LR = 0.0002"
echo "  - Epoch 243: LR ≈ 0.00017"
echo "  - Epoch 362: LR ≈ 0.0001"
echo "  - Epoch 600: LR ≈ 0.00001"
echo ""
echo "This is 47x higher than what worked at epoch 249!"
echo "Should see strong learning progress!"
echo "=========================================="
echo ""

/home/ubuntu/miniconda3/envs/imagenet/bin/python -u train.py \
  --config configs/resnet_strikes_back.yaml \
  --wandb_project imagenet-resnet50 \
  --wandb_name ${RUN_NAME} \
  --resume ${CHECKPOINT} \
  2>&1 | tee ${LOG_FILE}
