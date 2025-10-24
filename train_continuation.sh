#\!/bin/bash

cd /home/ubuntu/ImageNet-Full-training

RUN_NAME="RSB_A2_continuation_300to600_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/train_${RUN_NAME}.log"

echo "=========================================="
echo "RESNET STRIKES BACK A2 - CONTINUATION"
echo "=========================================="
echo "Run name: ${RUN_NAME}"
echo "Log file: ${LOG_FILE}"
echo ""
echo "CONTINUATION CONFIGURATION (Epochs 300-600):"
echo "  ✅ Starting from: epoch 286, 76.21% accuracy"
echo "  ✅ Peak LR: 1e-4 (50x lower than original)"
echo "  ✅ Warmup: 10 epochs (1e-6 → 1e-4)"
echo "  ✅ Cosine decay: 290 epochs to 1e-6"
echo "  ✅ Aggressive augmentation (RE 0.35, Mixup 0.3)"
echo "  ✅ Gradient accumulation (×2, effective batch 4096)"
echo "  ✅ Gradient clipping (1.0)"
echo "  ✅ Checkpoint dir: /mnt/checkpoints"
echo ""
echo "TARGET: 78-79% @ epoch 600"
echo "=========================================="
echo ""

/home/ubuntu/miniconda3/envs/imagenet/bin/python -u train.py \
  --config configs/resnet_continuation_300to600.yaml \
  --resume "checkpoints/resnet50-epoch=286-val/acc1=76.2100.ckpt" \
  --wandb_project imagenet-resnet50 \
  --wandb_name ${RUN_NAME} \
  2>&1 | tee ${LOG_FILE}
