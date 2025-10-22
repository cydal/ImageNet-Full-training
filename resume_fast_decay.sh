#\!/bin/bash

cd /home/ubuntu/ImageNet-Full-training

RUN_NAME="rsb_a2_fast_decay_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/train_${RUN_NAME}.log"
CHECKPOINT="checkpoints/epoch151_tmax100.ckpt"

echo "=========================================="
echo "RESUMING WITH FAST LR DECAY"
echo "=========================================="
echo "Run name: ${RUN_NAME}"
echo "Log file: ${LOG_FILE}"
echo ""
echo "FAST LR SCHEDULE (T_max=100):"
echo "  - Epoch 151: LR = 0.0198 (starting now)"
echo "  - Epoch 176: LR ≈ 0.0169 (25% through)"
echo "  - Epoch 201: LR ≈ 0.0099 (50% through)"
echo "  - Epoch 226: LR ≈ 0.0030 (75% through)"
echo "  - Epoch 251: LR ≈ 1e-6 (decay complete)"
echo "  - Epoch 252-600: LR ≈ 1e-6 (fine-tuning)"
echo ""
echo "LR will drop much faster now\!"
echo "=========================================="
echo ""

/home/ubuntu/miniconda3/envs/imagenet/bin/python -u train.py \
  --config configs/resnet_strikes_back.yaml \
  --wandb_project imagenet-resnet50 \
  --wandb_name ${RUN_NAME} \
  --resume ${CHECKPOINT} \
  2>&1 | tee ${LOG_FILE}
