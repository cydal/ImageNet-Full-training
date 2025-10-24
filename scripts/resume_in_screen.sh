#\!/bin/bash

cd /home/ubuntu/ImageNet-Full-training

RUN_NAME="rsb_a2_resumed_screen_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/train_${RUN_NAME}.log"
CHECKPOINT="checkpoints/last.ckpt"

echo "Starting training in screen session..."
echo "Run name: ${RUN_NAME}"
echo "Log file: ${LOG_FILE}"
echo "Checkpoint: ${CHECKPOINT}"
echo ""

/home/ubuntu/miniconda3/envs/imagenet/bin/python -u train.py \
  --config configs/resnet_strikes_back.yaml \
  --wandb_project imagenet-resnet50 \
  --wandb_name ${RUN_NAME} \
  --resume ${CHECKPOINT} \
  2>&1 | tee ${LOG_FILE}
