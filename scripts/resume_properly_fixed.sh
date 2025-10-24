#\!/bin/bash

cd /home/ubuntu/ImageNet-Full-training

RUN_NAME="rsb_a2_epoch123_fixed_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/train_${RUN_NAME}.log"
CHECKPOINT="checkpoints/epoch123_fixed_properly.ckpt"

echo "Resuming from epoch 123 with PROPERLY FIXED LR schedule"
echo "Run name: ${RUN_NAME}"
echo "Log file: ${LOG_FILE}"
echo ""
echo "LR SCHEDULE (CORRECT):"
echo "  - Epoch 123: LR = 0.020000 (starting now)"
echo "  - Epoch 242: LR ≈ 0.017083"
echo "  - Epoch 361: LR ≈ 0.010033"
echo "  - Epoch 599: LR ≈ 0.000001"
echo ""
echo "NO WARMUP - starts immediately at 0.02 and decays\!"
echo ""

/home/ubuntu/miniconda3/envs/imagenet/bin/python -u train.py \
  --config configs/resnet_strikes_back.yaml \
  --wandb_project imagenet-resnet50 \
  --wandb_name ${RUN_NAME} \
  --resume ${CHECKPOINT} \
  2>&1 | tee ${LOG_FILE}
