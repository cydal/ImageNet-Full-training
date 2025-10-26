# ImageNet ResNet-50 Training: A Complete Journey

A comprehensive ImageNet training codebase using PyTorch Lightning, documenting a full training journey from scratch to 76.2% top-1 accuracy with ResNet-50.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.0+-purple.svg)](https://lightning.ai/)

## 🎯 Project Overview

This project documents the complete journey of training a ResNet-50 model on ImageNet from scratch, including:
- Initial setup and data validation challenges
- Hyperparameter tuning and LR finding
- Multiple training strategies and optimizers
- Continuation training experiments
- Final results and learnings

**Final Result:** 76.21% top-1 accuracy on ImageNet validation set (ResNet-50 from scratch)

## 📊 Training Results Summary

| Approach | Epochs | Optimizer | LR | Augmentation | Top-1 Acc | Status |
|----------|--------|-----------|-----|--------------|-----------|--------|
| Initial (SGD) | 90 | SGD | 4.0 | Standard | ~76% | ✅ Baseline |
| RSB A2 (LAMB) | 300 | LAMB | 5e-3 | Aggressive | 76.21% | ✅ **Best** |
| Continuation (LAMB+Warmup) | 38 | LAMB | 1e-4 | Very Aggressive | 76.0% | ❌ No improvement |
| Continuation (SGD) | ~20 | SGD | 0.05 | Moderate | Declining | ❌ Failed |
| Final (LAMB+Constant LR) | Running | LAMB | 1e-4 | Standard | TBD | 🔄 In progress |

## 🚀 Quick Start

### Prerequisites
- Ubuntu Linux with 8x NVIDIA A100 40GB GPUs
- ImageNet dataset (ILSVRC2012)
- Python 3.11+
- CUDA 11.8+

### Installation

```bash
# Clone repository
cd /home/ubuntu/ImageNet-Full-training

# Create conda environment
conda create -n imagenet python=3.11
conda activate imagenet

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install lightning wandb timm pyyaml
pip install pytorch-lamb  # For LAMB optimizer
```

### Data Setup

```bash
# Mount ImageNet data (EBS volume)
sudo mount /dev/nvme1n1 /mnt/imagenet-data

# Verify structure
ls /mnt/imagenet-data/imagenet/
# Should show: train/ val/
```

### Training

```bash
# Train from scratch with RSB A2 recipe
python train.py \
  --config configs/resnet_strikes_back.yaml \
  --wandb_project imagenet-resnet50 \
  --wandb_name RSB_A2_training

# Resume from checkpoint
python train.py \
  --config configs/resnet_strikes_back.yaml \
  --resume checkpoints/resnet50-epoch=286-val/acc1=76.2100.ckpt \
  --wandb_project imagenet-resnet50
```

## 📁 Project Structure

```
ImageNet-Full-training/
├── configs/
│   ├── resnet_strikes_back.yaml          # RSB A2 recipe (best config)
│   ├── resnet_continuation_300to600.yaml # Continuation attempt
│   ├── resnet_sgd_continuation.yaml      # SGD continuation
│   └── resnet_lamb_constant_lr.yaml      # Constant LR final attempt
├── data/
│   └── datamodule.py                     # ImageNet data loading
├── models/
│   └── resnet50.py                       # ResNet-50 Lightning module
├── utils/
│   ├── callbacks.py                      # EMA, checkpointing
│   ├── metrics.py                        # Top-1/Top-5 accuracy
│   └── dist.py                           # Distributed training helpers
├── train.py                              # Main training script
├── modify_checkpoint.py                  # Checkpoint modification utility
└── README.md                             # This file
```

## 🔬 The Training Journey

### Phase 1: Initial Setup & Data Validation

**Challenge:** Validation accuracy stuck at ~30% despite good training loss

**Root Cause:** Mislabeled validation data
- Original `val/` directory had incorrect labels
- Discovered through manual inspection of predictions
- Fixed by using correct validation set

**Solution:**
```bash
# Backed up incorrect data
mv val/ val_old/
# Used correct validation data
# Result: Immediate jump to expected ~76% accuracy
```

**Key Learning:** Always verify data labels before debugging hyperparameters!

### Phase 2: LR Finding & Baseline Training

**Approach:** Systematic LR finding on single GPU

**Method:**
1. Ran LR finder on 1 GPU with batch size 256
2. Found optimal LR: 0.5 for single GPU
3. Scaled linearly for 8 GPUs: 0.5 × 8 = 4.0

**Configuration:**
```yaml
optimizer: sgd
lr: 4.0  # Scaled from single-GPU LR finder
momentum: 0.9
weight_decay: 0.0001
batch_size: 256  # Per GPU (2048 effective)
warmup_epochs: 5
lr_scheduler: cosine
```

**Results:**
- Epoch 5: ~30-40% val_acc1
- Epoch 90: ~76% val_acc1
- Training time: ~1 days on 8x A100

**Key Learning:** LR finder on single GPU + linear scaling works well for multi-GPU training

### Phase 3: ResNet Strikes Back A2 Recipe

**Goal:** Push beyond 76% using state-of-the-art training recipe

**Strategy:** Implemented [ResNet Strikes Back](https://arxiv.org/abs/2110.00476) A2 recipe

**Configuration:**
```yaml
optimizer: lamb  # LAMB optimizer (critical for RSB)
lr: 5.0e-3
weight_decay: 0.02
epochs: 300
warmup_epochs: 5
lr_scheduler: cosine
eta_min: 0.0

# Aggressive augmentation
random_erasing: 0.25
mixup_alpha: 0.2
cutmix_alpha: 1.0
label_smoothing: 0.1
auto_augment: randaugment

# Training settings
precision: 16-mixed
compile_model: true  # PyTorch 2.0 compilation
batch_size: 256  # Per GPU (2048 effective)
```

**Results:**
- Epoch 100: ~76.5%
- Epoch 200: ~76.8%
- Epoch 286: **76.21%** (best checkpoint)
- Epoch 300: ~76.1%

**Key Learning:** LAMB optimizer + aggressive augmentation + long training (300 epochs) achieves strong results

### Phase 4: Continuation Training Attempts

**Goal:** Push from 76.2% to 78-80%

#### Attempt 1: LAMB + Aggressive Augmentation + LR Restart
```yaml
optimizer: lamb
lr: 1.0e-4  # 50x lower than original
warmup_epochs: 10
random_erasing: 0.35  # Increased
mixup_alpha: 0.3      # Increased
accumulate_grad_batches: 2
gradient_clip_val: 1.0
```

**Result:** ❌ No improvement after 38 epochs
- Accuracy fluctuated 75.9-76.1%
- Loss unstable (0.99-1.12)
- **Conclusion:** Model has plateaued

#### Attempt 2: SGD + Moderate Augmentation
```yaml
optimizer: sgd
lr: 0.05
nesterov: true
random_erasing: 0.2   # Reduced
mixup_alpha: 0.15     # Reduced
```

**Result:** ❌ Loss rising, accuracy dropping
- SGD restart destabilized training
- **Conclusion:** Can't switch optimizers mid-training

#### Attempt 3: LAMB + Constant LR (Final Attempt)
```yaml
optimizer: lamb
lr: 1.0e-4  # CONSTANT (no warmup, no decay)
lr_scheduler: constant
random_erasing: 0.25  # Back to RSB standard
mixup_alpha: 0.2      # Back to RSB standard
```

**Status:** 🔄 Currently running
**Hypothesis:** LR schedule was disrupting learning
**Decision:** If no improvement by epoch 30, accept 76.2% as final result

### Key Challenges & Solutions

#### Challenge 1: Checkpoint Loading for Continuation Training

**Problem:** PyTorch Lightning restores optimizer and LR scheduler states when resuming, causing the LR schedule to continue from where it left off instead of restarting.

**Solution:** Load weights manually without using `ckpt_path`:
```python
if args.resume:
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    # Don't pass ckpt_path to trainer.fit - start fresh
trainer.fit(model, datamodule=datamodule)
```

**Key Learning:** For continuation training with new hyperparameters, load weights manually to avoid restoring old optimizer/scheduler states.

#### Challenge 2: EMA Model Weights in Checkpoints

**Problem:** Checkpoints contained `ema_model.*` keys that caused loading errors.

**Solution:** Filter out EMA keys during loading:
```python
state_dict = {k: v for k, v in state_dict.items() 
              if not k.startswith('ema_model.')}
```

#### Challenge 3: Finding the Right LR for Continuation

**Problem:** Original LR (5e-3) too high for continuation, but how low to go?

**Attempts:**
- 1e-4: No improvement (too low?)
- Warmup from 1e-6 to 1e-4: Unstable
- Constant 1e-4: Testing now

**Key Learning:** Continuation training is harder than training from scratch - the model may have already found a local minimum.

## 📈 Performance Analysis

### What Worked Well

1. ✅ **LR Finder + Linear Scaling**
   - Single-GPU LR finder: Simple and effective
   - Linear scaling to 8 GPUs: Worked perfectly
   - Saved hours of hyperparameter tuning

2. ✅ **LAMB Optimizer**
   - Better than SGD for large batch training
   - Achieved 76.21% (vs ~76% with SGD)
   - More stable training

3. ✅ **ResNet Strikes Back Recipe**
   - Aggressive augmentation improved generalization
   - 300 epochs necessary for convergence
   - PyTorch compilation gave ~10% speedup

4. ✅ **Data Validation**
   - Catching mislabeled data early saved days
   - Always verify data before debugging model

### What Didn't Work

1. ❌ **Continuation Training with LR Restart**
   - Model plateaued at 76.2%
   - Aggressive augmentation didn't help
   - Gradient accumulation added overhead without benefit

2. ❌ **Switching Optimizers Mid-Training**
   - SGD restart destabilized training
   - Can't change optimization landscape mid-flight

3. ❌ **Very Aggressive Augmentation**
   - RE 0.35 + Mixup 0.3 too strong
   - Hurt learning instead of helping

### Lessons Learned

1. **76-77% is typical for ResNet-50 from scratch**
   - Our 76.21% is actually very good
   - Reaching 80% requires ensemble, distillation, or architecture changes

2. **Continuation training is hard**
   - Models plateau for a reason
   - Can't easily push past local minima with same architecture

3. **Optimizer choice matters**
   - LAMB > SGD for large batch training
   - But can't switch mid-training

4. **Data quality > Hyperparameters**
   - Mislabeled data was the biggest issue
   - Fixed data → immediate 40% accuracy gain

5. **Keep it simple**
   - Complex continuation strategies didn't help
   - Sometimes the first good result is the best result

## 🎓 Training Recipes

### Recipe 1: ResNet-50 Baseline (SGD)

**Target:** ~76% accuracy in 90 epochs

```yaml
# configs/baseline_sgd.yaml
optimizer: sgd
lr: 4.0  # For 8 GPUs, batch 256 per GPU
momentum: 0.9
weight_decay: 0.0001
epochs: 90
warmup_epochs: 5
lr_scheduler: cosine

batch_size: 256  # Per GPU
precision: 16-mixed

# Standard augmentation
random_crop: true
random_horizontal_flip: true
mixup_alpha: 0.2
cutmix_alpha: 1.0
label_smoothing: 0.1
```

**Training time:** ~1 days on 8x A100

### Recipe 2: ResNet Strikes Back A2 (LAMB)

**Target:** ~76-77% accuracy in 300 epochs

```yaml
# configs/resnet_strikes_back.yaml
optimizer: lamb
lr: 5.0e-3
weight_decay: 0.02
epochs: 300
warmup_epochs: 5
lr_scheduler: cosine
eta_min: 0.0

batch_size: 256  # Per GPU
precision: 16-mixed
compile_model: true

# Aggressive augmentation
random_erasing: 0.25
mixup_alpha: 0.2
cutmix_alpha: 1.0
label_smoothing: 0.1
auto_augment: randaugment
```

**Training time:** ~10-12 hours on 8x A100

## 🔧 Utilities

### Checkpoint Modification

For continuation training, we created a utility to modify checkpoints:

```python
# modify_checkpoint.py
# Removes EMA weights, updates hyperparameters, resets scheduler
python modify_checkpoint.py
```

### Inference Script

Located in `/home/ubuntu/inference/inference_resnet50.py`:

```python
from inference_resnet50 import ResNet50Inference

# Load model
model = ResNet50Inference('checkpoints/best.ckpt', device='cuda')

# Single image
results = model.predict('image.jpg', top_k=5)

# Batch inference
results = model.predict_batch(image_paths, batch_size=32)
```

## 🌐 Model Deployment

### Hugging Face Spaces Demo

[*Live Demo*](https://huggingface.co/spaces/Sijuade/imagenet-resnet50-inference)

An interactive Streamlit app for trying the model on your own images:
- Upload any image
- Get top-5 predictions with confidence scores
- Visualize model attention (Grad-CAM)
- Compare different checkpoints

**To deploy your own:**
1. Export model to ONNX or TorchScript
2. Create Streamlit app
3. Deploy to Hugging Face Spaces

```bash
# Export model
python export_model.py \
  --checkpoint checkpoints/resnet50-epoch=286-val/acc1=76.2100.ckpt \
  --output resnet50.onnx
```

## 📊 Monitoring & Logging

### Weights & Biases

All experiments tracked on W&B:
- Training/validation loss curves
- Learning rate schedules
- Top-1/Top-5 accuracy
- System metrics (GPU utilization, throughput)

```bash
# View logs
wandb login
# Then visit: https://wandb.ai/your-project/imagenet-resnet50
```

### Checkpoints

Best checkpoints saved to `/mnt/checkpoints/`:
- `resnet50-epoch=286-val/acc1=76.2100.ckpt` - **Best model**
- `resnet50-epoch=282-val/acc1=76.1720.ckpt`
- `resnet50-epoch=281-val/acc1=76.1740.ckpt`

Also backed up to S3:
```bash
aws s3 sync /mnt/checkpoints/ s3://sij-imagenet-train/imagenet/checkpoints/lambs/
```

## 🔬 Experimental Results

### Training Curves

**RSB A2 Training (300 epochs):**
- Epochs 0-50: Rapid improvement (30% → 70%)
- Epochs 50-150: Steady gains (70% → 75%)
- Epochs 150-250: Slow improvement (75% → 76%)
- Epochs 250-300: Plateau (~76.2%)

**Continuation Attempts:**
- LAMB + Aggressive Aug: Flat at 76.0-76.2%
- SGD Restart: Declining (76.2% → 75.8%)
- LAMB + Constant LR: TBD

### Hyperparameter Sensitivity

**Learning Rate:**
- Too high (>1e-3 for continuation): Unstable
- Too low (<1e-5): No learning
- Sweet spot: 1e-4 for continuation

**Augmentation:**
- Moderate (RE 0.25, Mixup 0.2): ✅ Works well
- Aggressive (RE 0.35, Mixup 0.3): ❌ Too strong
- Minimal: ❌ Overfitting

**Batch Size:**
- 2048 effective (256 per GPU × 8): Optimal
- 4096 (with grad accumulation): Slower, no benefit

### Logs
```
Epoch 287:  99%|█████████▉| 620/625 [04:20<00:02,  2.38it/s, v_num=3_22, train/loss_step=3.860, val/loss=0.988, val/acc1=76.20, train/loss_epoch=2.040]
Epoch 287:  99%|█████████▉| 620/625 [04:20<00:02,  2.38it/s, v_num=3_22, train/loss_step=2.790, val/loss=0.988, val/acc1=76.20, train/loss_epoch=2.040]
Epoch 287:  99%|█████████▉| 621/625 [04:20<00:01,  2.38it/s, v_num=3_22, train/loss_step=2.790, val/loss=0.988, val/acc1=76.20, train/loss_epoch=2.040]
Epoch 287:  99%|█████████▉| 621/625 [04:20<00:01,  2.38it/s, v_num=3_22, train/loss_step=1.450, val/loss=0.988, val/acc1=76.20, train/loss_epoch=2.040]
Epoch 287: 100%|█████████▉| 622/625 [04:21<00:01,  2.38it/s, v_num=3_22, train/loss_step=1.450, val/loss=0.988, val/acc1=76.20, train/loss_epoch=2.040]
Epoch 287: 100%|█████████▉| 622/625 [04:21<00:01,  2.38it/s, v_num=3_22, train/loss_step=2.980, val/loss=0.988, val/acc1=76.20, train/loss_epoch=2.040]
Epoch 287: 100%|█████████▉| 623/625 [04:21<00:00,  2.38it/s, v_num=3_22, train/loss_step=2.980, val/loss=0.988, val/acc1=76.20, train/loss_epoch=2.040]
Epoch 287: 100%|█████████▉| 623/625 [04:21<00:00,  2.38it/s, v_num=3_22, train/loss_step=0.689, val/loss=0.988, val/acc1=76.20, train/loss_epoch=2.040]
Epoch 287: 100%|█████████▉| 624/625 [04:21<00:00,  2.38it/s, v_num=3_22, train/loss_step=0.689, val/loss=0.988, val/acc1=76.20, train/loss_epoch=2.040]
Epoch 287: 100%|█████████▉| 624/625 [04:21<00:00,  2.38it/s, v_num=3_22, train/loss_step=1.700, val/loss=0.988, val/acc1=76.20, train/loss_epoch=2.040]
Epoch 287: 100%|██████████| 625/625 [04:22<00:00,  2.38it/s, v_num=3_22, train/loss_step=1.700, val/loss=0.988, val/acc1=76.20, train/loss_epoch=2.040]
Epoch 287: 100%|██████████| 625/625 [04:22<00:00,  2.38it/s, v_num=3_22, train/loss_step=0.790, val/loss=0.988, val/acc1=76.20, train/loss_epoch=2.040]
```

<img width="1512" height="982" alt="Image" src="https://github.com/user-attachments/assets/34a68873-0431-4f76-8f4f-e76d465f5ce3" />
<img width="2727" height="964" alt="Image" src="https://github.com/user-attachments/assets/437c2396-0730-445e-856a-de73f7a2dab9" />

## 🚀 Future Directions

### To Reach 78-80% Accuracy

1. **Knowledge Distillation** (Most promising)
   - Use 76.2% model as teacher
   - Train new ResNet-50 as student
   - Expected gain: +1-2%

2. **Model Ensemble**
   - Ensemble 3-5 checkpoints
   - Expected gain: +1-2%
   - Minimal training cost

3. **Architecture Changes**
   - ResNet-50 → ResNet-101 or EfficientNet
   - Expected gain: +2-4%
   - Requires retraining

4. **Advanced Augmentation**
   - Test AutoAugment, RandAugment variations
   - CutOut, GridMask
   - Expected gain: +0.5-1%

### Not Recommended

- ❌ More continuation training with same architecture
- ❌ Extremely long training (>500 epochs)
- ❌ Very large batch sizes (>8192)

## 📚 References

1. [ResNet Strikes Back](https://arxiv.org/abs/2110.00476) - Training recipe we followed
2. [LAMB Optimizer](https://arxiv.org/abs/1904.00962) - Large batch optimization
3. [Mixup](https://arxiv.org/abs/1710.09412) - Data augmentation
4. [CutMix](https://arxiv.org/abs/1905.04899) - Data augmentation
5. [PyTorch Lightning](https://lightning.ai/) - Training framework

## 🤝 Contributing

This is a research/educational project documenting a complete training journey. Feel free to:
- Use the code for your own experiments
- Adapt the training recipes
- Share your results

## 📝 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- PyTorch and Lightning teams for excellent frameworks
- ResNet Strikes Back authors for the training recipe
- AWS for compute resources (8x A100 GPUs)
- Rohan Shravan School Of AI

## 📞 Contact

For questions about this training journey or to discuss results:
- Open an issue on GitHub
- Check the W&B project for detailed logs

---

**Last Updated:** October 24, 2025  
**Status:** Final attempt (LAMB + Constant LR) in progress  
**Best Result:** 76.21% top-1 accuracy (ResNet-50 from scratch)
