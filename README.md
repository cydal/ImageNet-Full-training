# ImageNet Training with PyTorch Lightning

A production-ready ImageNet training codebase using PyTorch Lightning, designed for efficient single-node and multi-node distributed training.

## Features

- **PyTorch Lightning**: Clean, modular training code with automatic distributed training support
- **Flexible Configuration**: YAML-based configs with easy overrides for different experiment settings
- **Multi-Node Support**: Built-in support for distributed training across multiple nodes
- **Modern Augmentations**: AutoAugment, Mixup, CutMix, and label smoothing
- **EMA Support**: Exponential Moving Average for better model performance
- **FSx Integration**: Scripts for mounting and using AWS FSx for high-performance data loading
- **Smoke Testing**: Tiny subset creation for quick validation

## Project Structure

```
├─ configs/
│  ├─ base.yaml                # shared defaults
│  ├─ tiny.yaml                # overrides: data_root, img_size, batch_size
│  ├─ full.yaml                # overrides for full run
├─ data/
│  └─ datamodule.py            # LightningDataModule (ImageNet, tiny/med/full)
├─ models/
│  └─ resnet50.py              # model factory, loss, metrics (wraps torchvision)
├─ train.py                    # Lightning Trainer entrypoint
├─ eval.py                     # evaluate checkpoint on val only
├─ utils/
│  ├─ callbacks.py             # ckpt, LR monitor, EMA (optional)
│  ├─ metrics.py               # top1/top5
│  └─ dist.py                  # multi-node launch helpers (env read)
├─ scripts/
│  ├─ env_setup.sh             # apt + pip deps; optional DALI
│  ├─ make_tiny_subset.py      # (symlink subset) for smoke tests
│  ├─ mount_fsx.sh             # mount FSx at /fsx
│  ├─ launch_single.sh         # single node torchrun
│  └─ launch_multi.sh          # multi-node torchrun (reads HOSTS from file)
├─ Makefile                    # one-liners
├─ requirements.txt            # pytorch, lightning, torchvision, timm, wandb
└─ README.md
```

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
make setup

# Or just install Python packages
make install
```

### 2. Mount FSx (Optional)

If using AWS FSx for data storage:

```bash
export FSX_DNS_NAME=fs-xxxxx.fsx.us-east-1.amazonaws.com
make mount-fsx
```

### 3. Create Tiny Subset for Testing

```bash
make tiny-subset
```

This creates a small subset at `/fsx/imagenet-tiny` with 10 classes for quick smoke tests.

### 4. Train

**Smoke test on tiny subset:**
```bash
make train-tiny
```

**Single-node training:**
```bash
make train-single
# Or with custom config
CONFIG=configs/full.yaml make train-single
```

**Multi-node training:**
```bash
# Create hosts.txt with one hostname per line
echo "node1" > hosts.txt
echo "node2" >> hosts.txt

# Run on each node
make train-multi
```

### 5. Evaluate

```bash
make eval CHECKPOINT=checkpoints/resnet50-epoch=89.ckpt
```

## Configuration

The training is configured through YAML files in `configs/`:

- **`base.yaml`**: Default settings for all experiments
- **`tiny.yaml`**: Quick smoke test with small dataset and model
- **`full.yaml`**: Production settings for full ImageNet training

### Key Configuration Options

```yaml
# Data
data_root: /fsx/imagenet
batch_size: 256
num_workers: 8

# Model
model_name: resnet50
num_classes: 1000

# Training
epochs: 90
lr: 0.1
optimizer: sgd
lr_scheduler: cosine
warmup_epochs: 5

# Augmentation
auto_augment: imagenet
mixup_alpha: 0.2
cutmix_alpha: 1.0
label_smoothing: 0.1

# Mixed precision
precision: 16-mixed

# EMA
use_ema: true
ema_decay: 0.9999
```

## Command-Line Usage

### Training

```bash
# Basic training
python train.py --config configs/base.yaml

# Override config values
python train.py --config configs/base.yaml \
    --data_root /custom/path \
    --batch_size 512 \
    --lr 0.2 \
    --epochs 100

# Resume from checkpoint
python train.py --config configs/base.yaml \
    --resume checkpoints/last.ckpt

# Disable W&B logging
python train.py --config configs/base.yaml --no_wandb
```

### Evaluation

```bash
python eval.py \
    --checkpoint checkpoints/resnet50-epoch=89.ckpt \
    --config configs/base.yaml
```

## Distributed Training

### Single Node, Multiple GPUs

```bash
# Automatic GPU detection
bash scripts/launch_single.sh

# Or specify number of GPUs
NUM_GPUS=8 bash scripts/launch_single.sh
```

### Multiple Nodes

1. Create a `hosts.txt` file with one hostname per line:
```
node1.compute.internal
node2.compute.internal
node3.compute.internal
node4.compute.internal
```

2. Launch on each node:
```bash
bash scripts/launch_multi.sh
```

The script automatically determines the node rank based on the hostname.

## Data Format

Expected ImageNet directory structure:

```
/fsx/imagenet/
├── train/
│   ├── n01440764/
│   │   ├── n01440764_10026.JPEG
│   │   └── ...
│   ├── n01443537/
│   │   └── ...
│   └── ...
└── val/
    ├── n01440764/
    │   ├── ILSVRC2012_val_00000293.JPEG
    │   └── ...
    └── ...
```

## Performance Tips

1. **Use FSx for Lustre**: Provides high-throughput parallel file system for data loading
2. **Increase num_workers**: Set to 8-12 per GPU for optimal data loading
3. **Enable persistent_workers**: Reduces worker initialization overhead
4. **Use mixed precision**: `precision: 16-mixed` for faster training with minimal accuracy loss
5. **Enable EMA**: Improves final model accuracy with minimal overhead
6. **Tune batch size**: Larger batches generally train faster but may require learning rate adjustment

## Monitoring

Training metrics are logged to:
- **Weights & Biases**: Real-time monitoring and experiment tracking
- **CSV logs**: Local CSV files in `logs/csv_logs/`
- **TensorBoard**: Compatible with PyTorch Lightning's logger

View logs:
```bash
# W&B (if enabled)
wandb login
# Then view at https://wandb.ai

# TensorBoard
tensorboard --logdir logs/
```

## Checkpoints

Checkpoints are saved to `checkpoints/` with the following naming:
```
resnet50-epoch=XX-val_acc1=0.XXXX.ckpt
```

The best `save_top_k` checkpoints are kept based on validation accuracy.

## Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Reduce `num_workers`
- Use gradient accumulation (add to config)

### Slow Data Loading
- Increase `num_workers`
- Enable `persistent_workers`
- Use FSx or faster storage
- Consider NVIDIA DALI (install via `scripts/env_setup.sh`)

### Multi-Node Issues
- Ensure all nodes can communicate (check firewall)
- Verify `MASTER_ADDR` and `MASTER_PORT` are correct
- Check that all nodes have the same code and environment

## License

MIT License

## Citation

If you use this codebase, please cite:

```bibtex
@misc{imagenet-lightning,
  title={ImageNet Training with PyTorch Lightning},
  year={2024},
  url={https://github.com/yourusername/imagenet}
}
```
