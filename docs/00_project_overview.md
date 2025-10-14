# ImageNet Training Project Overview

## Goal
Train ResNet50 from scratch on full ImageNet-1K dataset using PyTorch Lightning with multi-node distributed training capability.

## Development Phases

### Phase 1: Local Development (Current)
- **Data Location**: `/data2/imagenet` (local storage)
- **Setup**: Single node, multi-GPU
- **Goal**: Validate training pipeline, data loading, and model implementation

### Phase 2: Production Deployment
- **Data Location**: `/fsx/imagenet` (AWS FSx for Lustre)
- **Setup**: Multi-node, multi-GPU cluster
- **Goal**: Full-scale training with optimal throughput

## Technical Stack
- **Framework**: PyTorch Lightning 2.0+
- **Model**: ResNet50 (torchvision)
- **Data**: ImageNet-1K (ILSVRC2012)
- **Distributed**: DDP (DistributedDataParallel)
- **Mixed Precision**: AMP (Automatic Mixed Precision)
- **Logging**: Weights & Biases, CSV logs

## Target Metrics
- **Top-1 Accuracy**: ~76.1% (ResNet50 baseline)
- **Top-5 Accuracy**: ~92.9%
- **Training Time**: ~90 epochs
- **Throughput**: Target 1000+ images/sec per GPU

## Project Structure
```
imagenet/
├── configs/          # YAML configuration files
├── data/            # Data loading and preprocessing
├── models/          # Model definitions and wrappers
├── utils/           # Utilities (callbacks, metrics, distributed)
├── scripts/         # Shell scripts for setup and launching
├── docs/            # Development documentation
├── train.py         # Training entrypoint
├── eval.py          # Evaluation script
└── Makefile         # Convenient commands
```

## Current Status
- ✅ Project structure created
- ✅ Configuration system implemented
- ✅ Data module skeleton created
- ⏳ Data module testing (in progress)
- ⏳ Model implementation validation
- ⏳ Training loop testing
- ⏳ Multi-GPU validation
- ⏳ Multi-node setup

## Development Workflow
1. **Data Module**: Verify data loading and preprocessing
2. **Model**: Test forward pass and loss computation
3. **Training**: Single GPU smoke test
4. **Scaling**: Multi-GPU on single node
5. **Production**: Multi-node distributed training

## Key Configuration Files
- `configs/base.yaml`: Default settings for all experiments
- `configs/tiny.yaml`: Quick smoke tests (10 classes)
- `configs/full.yaml`: Full training configuration
- `configs/local.yaml`: Local development settings (to be created)

## Documentation Files
- `00_project_overview.md`: This file
- `01_data_module.md`: Data loading documentation
- `02_model.md`: Model architecture and training (to be created)
- `03_distributed.md`: Multi-node setup (to be created)
- `04_benchmarks.md`: Performance benchmarks (to be created)
