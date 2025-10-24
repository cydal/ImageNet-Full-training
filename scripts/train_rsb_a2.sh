#\!/bin/bash

cd /home/ubuntu/ImageNet-Full-training

RUN_NAME="RSB_A2_FULL_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/train_${RUN_NAME}.log"

echo "=========================================="
echo "RESNET STRIKES BACK A2 - FULL RECIPE"
echo "=========================================="
echo "Run name: ${RUN_NAME}"
echo "Log file: ${LOG_FILE}"
echo ""
echo "COMPLETE RSB A2 CONFIGURATION:"
echo "  ✅ LAMB optimizer (LR=5e-3, WD=0.02)"
echo "  ✅ BCE loss with soft targets"
echo "  ✅ RandAugment (M≈7-9)"
echo "  ✅ Random Erasing (p=0.25)"
echo "  ✅ Mixup (α=0.2) + CutMix (α=1.0)"
echo "  ✅ EMA (decay=0.9999)"
echo "  ✅ Batch 2048 (8×256)"
echo "  ✅ 300 epochs, 5 warmup, cosine to 0"
echo ""
echo "TARGET: ~79.8% @ 300 epochs"
echo "=========================================="
echo ""

/home/ubuntu/miniconda3/envs/imagenet/bin/python -u train.py \
  --config configs/resnet_strikes_back.yaml \
  --wandb_project imagenet-resnet50 \
  --wandb_name ${RUN_NAME} \
  2>&1 | tee ${LOG_FILE}
