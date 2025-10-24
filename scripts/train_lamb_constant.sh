#!/bin/bash

cd /home/ubuntu/ImageNet-Full-training

RUN_NAME="LAMB_constant_LR_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/train_${RUN_NAME}.log"

echo "=========================================="
echo "RESNET-50 FINAL ATTEMPT - LAMB + CONSTANT LR"
echo "=========================================="
echo "Run name: ${RUN_NAME}"
echo "Log file: ${LOG_FILE}"
echo ""
echo "FINAL STRATEGY:"
echo "  ✅ Optimizer: LAMB (what got us to 76.2%)"
echo "  ✅ Augmentation: RSB A2 standard (RE 0.25, Mixup 0.2)"
echo "  ✅ LR: 1e-4 CONSTANT (no warmup, no decay)"
echo "  ✅ Starting from: epoch 286, 76.21% accuracy"
echo "  ✅ Total: 100 epochs"
echo "  ✅ Checkpoint dir: /mnt/checkpoints"
echo ""
echo "HYPOTHESIS: LR schedule was the problem, not optimizer/augmentation"
echo "DECISION: If no improvement by epoch 30, accept 76.2% as final"
echo "=========================================="
echo ""

/home/ubuntu/miniconda3/envs/imagenet/bin/python -u train.py \
  --config configs/resnet_lamb_constant_lr.yaml \
  --resume "checkpoints/resnet50-epoch=286-val/acc1=76.2100.ckpt" \
  --wandb_project imagenet-resnet50 \
  --wandb_name ${RUN_NAME} \
  2>&1 | tee ${LOG_FILE}
