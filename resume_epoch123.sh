#\!/bin/bash

cd /home/ubuntu/ImageNet-Full-training

RUN_NAME="rsb_a2_from_epoch123_lr0.02_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/train_${RUN_NAME}.log"
CHECKPOINT="checkpoints/epoch123_lr0.02_t477_modified.ckpt"

echo "Resuming training from epoch 123..."
echo "Run name: ${RUN_NAME}"
echo "Log file: ${LOG_FILE}"
echo "Checkpoint: ${CHECKPOINT}"
echo ""
echo "LR SCHEDULE:"
echo "  - Starting LR: 0.02 (at epoch 123)"
echo "  - Decay over: 477 epochs (epoch 123 → 600)"
echo "  - Minimum LR: 1e-6 (never reaches 0)"
echo "  - Epoch 600: LR ≈ 1e-6"
echo ""

/home/ubuntu/miniconda3/envs/imagenet/bin/python -u train.py \
  --config configs/resnet_strikes_back.yaml \
  --wandb_project imagenet-resnet50 \
  --wandb_name ${RUN_NAME} \
  --resume ${CHECKPOINT} \
  2>&1 | tee ${LOG_FILE}
