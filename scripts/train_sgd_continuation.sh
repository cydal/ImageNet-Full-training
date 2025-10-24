#!/bin/bash

cd /home/ubuntu/ImageNet-Full-training

RUN_NAME="SGD_continuation_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/train_${RUN_NAME}.log"

echo "=========================================="
echo "RESNET-50 SGD CONTINUATION"
echo "=========================================="
echo "Run name: ${RUN_NAME}"
echo "Log file: ${LOG_FILE}"
echo ""
echo "NEW STRATEGY - BACK TO BASICS:"
echo "  ✅ Optimizer: SGD with Nesterov momentum"
echo "  ✅ Starting from: epoch 286, 76.21% accuracy"
echo "  ✅ LR: 0.05 (conservative restart)"
echo "  ✅ Warmup: 5 epochs"
echo "  ✅ Total: 100 epochs (fast iteration)"
echo "  ✅ Moderate augmentation (RE 0.2, Mixup 0.15)"
echo "  ✅ No gradient accumulation (SGD likes frequent updates)"
echo "  ✅ Checkpoint dir: /mnt/checkpoints"
echo ""
echo "TARGET: 77-78% within 100 epochs"
echo "DECISION POINT: Epoch 30 - if no improvement, pivot to AdamW"
echo "=========================================="
echo ""

/home/ubuntu/miniconda3/envs/imagenet/bin/python -u train.py \
  --config configs/resnet_sgd_continuation.yaml \
  --resume "checkpoints/resnet50-epoch=286-val/acc1=76.2100.ckpt" \
  --wandb_project imagenet-resnet50 \
  --wandb_name ${RUN_NAME} \
  2>&1 | tee ${LOG_FILE}
