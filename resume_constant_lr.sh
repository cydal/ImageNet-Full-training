#!/bin/bash

cd /home/ubuntu/ImageNet-Full-training

RUN_NAME="rsb_a2_constant_lr_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/train_${RUN_NAME}.log"
CHECKPOINT="checkpoints/epoch249_constant_lr3e-5.ckpt"

echo "=========================================="
echo "RESUMING WITH CONSTANT LR"
echo "=========================================="
echo "Run name: ${RUN_NAME}"
echo "Log file: ${LOG_FILE}"
echo "Checkpoint: ${CHECKPOINT}"
echo ""
echo "TRAINING CONFIGURATION:"
echo "  - Starting epoch: 249"
echo "  - Starting accuracy: 71.8%"
echo "  - Constant LR: 3e-5 (0.00003)"
echo "  - No LR decay - stays constant until epoch 600"
echo "  - Remaining epochs: 351"
echo ""
echo "This stable LR should prevent divergence!"
echo "=========================================="
echo ""

/home/ubuntu/miniconda3/envs/imagenet/bin/python -u train.py \
  --config configs/resnet_strikes_back.yaml \
  --wandb_project imagenet-resnet50 \
  --wandb_name ${RUN_NAME} \
  --resume ${CHECKPOINT} \
  2>&1 | tee ${LOG_FILE}
