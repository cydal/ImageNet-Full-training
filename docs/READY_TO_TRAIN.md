# Ready to Train! 🚀

## Current Status

✅ **Training configurations created** (ResNet Strikes Back recipe)  
✅ **W&B setup script ready**  
✅ **Helper scripts created**  
✅ **Documentation complete**  
✅ **FSx mounted** at `/fsx` with ImageNet data at `/fsx/ns1/`  
✅ **GPU available** (NVIDIA A10G, 23GB)

## Before You Start

### 1. Activate Conda Environment

```bash
conda activate imagenet
```

### 2. Verify Data Paths

Your ImageNet data should be at:
```
/fsx/ns1/train/  # 1000 class directories
/fsx/ns1/val/    # 1000 class directories
```

Verify:
```bash
ls -la /fsx/ns1/train/ | head
ls -la /fsx/ns1/val/ | head
```

### 3. Setup W&B (Recommended)

```bash
cd /home/ubuntu/ImageNet-Full-training
./setup_wandb.sh
```

Get your API key from: https://wandb.ai/authorize

## Start Training

### Option 1: With W&B Logging (Recommended)

```bash
cd /home/ubuntu/ImageNet-Full-training
conda activate imagenet

python train.py \
    --config configs/single_gpu_full.yaml \
    --wandb_project imagenet-resnet50 \
    --wandb_name single-gpu-run-1
```

### Option 2: Without W&B

```bash
cd /home/ubuntu/ImageNet-Full-training
conda activate imagenet

python train.py \
    --config configs/single_gpu_full.yaml \
    --no_wandb
```

### Option 3: Adjust Batch Size (if OOM)

```bash
python train.py \
    --config configs/single_gpu_full.yaml \
    --batch_size 64 \
    --lr 0.125 \
    --no_wandb
```

## Training Configuration

**Single GPU Full Training:**
- **Epochs:** 100
- **Batch size:** 128 (adjust if OOM)
- **Learning rate:** 0.25 (scaled for batch size)
- **Augmentation:** Mixup + CutMix + Label Smoothing
- **Precision:** FP16 (mixed precision)
- **Expected accuracy:** ~77-78% top-1
- **Training time:** ~7-10 days on A10G

## What to Expect

### First Few Minutes
```
GPU available: True (cuda), used: 1
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Using 1000 classes for training
Created subset: 1281167 samples from 1000 classes  # Full training set
Created subset: 50000 samples from 1000 classes    # Full validation set

Epoch 0:   0%|          | 0/10009 [00:00<?, ?it/s]
```

### During Training
```
Epoch 0:  50%|█████     | 5000/10009 [1:07:30<1:07:30, 1.23it/s, loss=6.91]
Epoch 0: 100%|██████████| 10009/10009 [2:15:23<00:00, 1.23it/s, loss=6.85]
Validation: 100%|██████████| 391/391 [05:12<00:00, 1.25it/s]

Epoch 0: val/acc1=0.0123, val/acc5=0.0567, val/loss=6.89
```

### Expected Progress

| Epoch | Hours | Train Loss | Val Acc@1 | Val Acc@5 |
|-------|-------|------------|-----------|-----------|
| 1     | 2.5   | 6.5        | 1-2%      | 5-10%     |
| 10    | 25    | 4.5        | 30%       | 55%       |
| 25    | 62    | 3.0        | 50%       | 75%       |
| 50    | 125   | 2.0        | 65%       | 85%       |
| 100   | 250   | 1.5        | 77%       | 93%       |

## Monitoring

### W&B Dashboard
- URL will be printed when training starts
- Example: `https://wandb.ai/your-username/imagenet-resnet50/runs/xxxxx`

### Terminal
```bash
# In another terminal, monitor progress
tail -f logs/train_*.log
```

### GPU Usage
```bash
# Monitor GPU utilization
watch -n 1 nvidia-smi
```

### Checkpoints
```bash
# Check saved checkpoints
ls -lh checkpoints/
```

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size to 64
python train.py \
    --config configs/single_gpu_full.yaml \
    --batch_size 64 \
    --lr 0.125 \
    --no_wandb
```

### Training Crashes
```bash
# Resume from last checkpoint
python train.py \
    --config configs/single_gpu_full.yaml \
    --resume checkpoints/last.ckpt
```

### Slow Data Loading
```bash
# Increase workers
python train.py \
    --config configs/single_gpu_full.yaml \
    --num_workers 16
```

## After Training Starts Successfully

1. ✓ **Verify W&B dashboard** - Check metrics are logging
2. ✓ **Monitor first epoch** - Should complete in ~2.5 hours
3. ✓ **Check validation accuracy** - Should be >1% after epoch 1
4. ✓ **Verify checkpoints** - Should save to `checkpoints/`
5. → **Let it run** - Training takes ~7-10 days

## Next Steps

Once single GPU training is stable (after 5-10 epochs):

1. **Verify metrics look good** in W&B
2. **Test checkpoint resume** works
3. **Plan multi-GPU scaling** (4-8 GPUs)
4. **Plan multi-node scaling** (multiple instances)

## Quick Command Reference

```bash
# Activate environment
conda activate imagenet

# Start training
python train.py --config configs/single_gpu_full.yaml

# Start with W&B
python train.py \
    --config configs/single_gpu_full.yaml \
    --wandb_project imagenet-resnet50 \
    --wandb_name run-1

# Resume training
python train.py \
    --config configs/single_gpu_full.yaml \
    --resume checkpoints/last.ckpt

# Monitor GPU
nvidia-smi

# Check checkpoints
ls -lh checkpoints/

# View logs
tail -f logs/train_*.log
```

## Documentation

- **Quick start:** `START_TRAINING.md`
- **Detailed guide:** `docs/TRAINING_GUIDE.md`
- **FSx setup:** `docs/FSX_SETUP.md`
- **Troubleshooting:** `docs/TRAINING_FIX.md`

---

**You're all set! Start training when ready.** 🎯
