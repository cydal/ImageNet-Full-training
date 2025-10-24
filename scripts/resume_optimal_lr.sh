#!/bin/bash

cd /home/ubuntu/ImageNet-Full-training

RUN_NAME="rsb_a2_optimal_lr6e-6_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/train_${RUN_NAME}.log"
CHECKPOINT="checkpoints/epoch249_constant_lr6e-6.ckpt"

echo "=========================================="
echo "RESUMING WITH OPTIMAL LR FROM RANGE TEST"
echo "=========================================="
echo "Run name: ${RUN_NAME}"
echo "Log file: ${LOG_FILE}"
echo "Checkpoint: ${CHECKPOINT}"
echo ""
echo "TRAINING CONFIGURATION:"
echo "  - Starting epoch: 249"
echo "  - Starting accuracy: 71.8%"
echo "  - Constant LR: 6e-6 (0.000006)"
echo "  - Found via LR range test (steepest descent)"
echo "  - This is the OPTIMAL LR for this stage!"
echo "  - Remaining epochs: 351"
echo ""
echo "This should show steady loss reduction!"
echo "=========================================="
echo ""

/home/ubuntu/miniconda3/envs/imagenet/bin/python -u train.py \
  --config configs/resnet_strikes_back.yaml \
  --wandb_project imagenet-resnet50 \
  --wandb_name ${RUN_NAME} \
  --resume ${CHECKPOINT} \
  2>&1 | tee ${LOG_FILE}
