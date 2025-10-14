# Documentation Index

This directory contains development documentation for the ImageNet training project.

## Quick Links

### Getting Started
- **[Getting Started Guide](03_getting_started.md)** - Start here! Quick setup and first training run

### Core Documentation
1. **[Project Overview](00_project_overview.md)** - Goals, architecture, and roadmap
2. **[Data Module](01_data_module.md)** - Data loading, preprocessing, and augmentation
3. **[Local Development](02_local_development.md)** - Development workflow and troubleshooting

### Coming Soon
- `04_model_architecture.md` - ResNet50 implementation details
- `05_training_loop.md` - Training configuration and hyperparameters
- `06_distributed_training.md` - Multi-node setup and optimization
- `07_benchmarks.md` - Performance metrics and comparisons

## Quick Start

```bash
# 1. Install dependencies
make install

# 2. Run quick test
make quick-test

# 3. Train locally
make train-local
```

## Current Status

### ✅ Completed
- Project structure setup
- Configuration system (YAML-based)
- Data module implementation
- Model wrapper (ResNet50 + Lightning)
- Training and evaluation scripts
- Local development configuration
- Documentation framework

### 🔄 In Progress
- Data module testing and validation
- Performance benchmarking
- Multi-GPU testing

### ⏳ Planned
- Multi-node distributed training
- FSx integration
- Production deployment
- Full 90-epoch training run

## Documentation Structure

```
docs/
├── README.md                    # This file
├── 00_project_overview.md       # High-level overview
├── 01_data_module.md            # Data loading details
├── 02_local_development.md      # Development guide
└── 03_getting_started.md        # Quick start guide
```

## Contributing to Documentation

When adding new features or making changes:
1. Update relevant documentation files
2. Add new sections to existing docs or create new files
3. Update this README with links to new documentation
4. Keep documentation in sync with code

## Key Concepts

### Configuration Hierarchy
```
base.yaml (defaults)
  ├── local.yaml (local development)
  ├── tiny.yaml (smoke tests)
  └── full.yaml (production)
```

### Data Paths
- **Local**: `/data2/imagenet` (current)
- **FSx**: `/fsx/imagenet` (future)

### Training Modes
- **Quick test**: 1 epoch, no logging
- **Local dev**: 10 epochs, local data
- **Production**: 90 epochs, FSx data, multi-node

## Useful Links

- [PyTorch Lightning Docs](https://lightning.ai/docs/pytorch/stable/)
- [ImageNet Dataset](https://www.image-net.org/)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [Weights & Biases](https://wandb.ai/)
