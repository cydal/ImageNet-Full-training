# Data Module Documentation

## Overview
The ImageNet data module is built using PyTorch Lightning's `LightningDataModule` class, providing a clean interface for data loading and preprocessing.

## Dataset Location
- **Local Development**: `/data2/imagenet`
- **Production (FSx)**: `/fsx/imagenet` (to be configured later)

## Dataset Structure
```
/data2/imagenet/
├── train/              # 1000 class directories
│   ├── n01440764/      # Class synset ID
│   │   ├── n01440764_10026.JPEG
│   │   ├── n01440764_10027.JPEG
│   │   └── ...
│   ├── n01443537/
│   └── ...
└── val/                # 1000 class directories (same structure)
    ├── n01440764/
    └── ...
```

## Data Statistics
- **Training images**: ~1.28M images
- **Validation images**: 50K images (50 per class)
- **Number of classes**: 1000
- **Image format**: JPEG

## Data Augmentation

### Training Augmentation Pipeline
1. **RandomResizedCrop(224)**: Random crop with scale (0.08, 1.0)
2. **RandomHorizontalFlip**: 50% probability
3. **Optional AutoAugment**: ImageNet policy
4. **ToTensor**: Convert to tensor
5. **Normalize**: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

### Validation Preprocessing
1. **Resize(256)**: Resize shorter side to 256
2. **CenterCrop(224)**: Center crop to 224x224
3. **ToTensor**: Convert to tensor
4. **Normalize**: Same as training

## Advanced Augmentation (Optional)
- **Mixup**: Mix two images and labels (alpha=0.2)
- **CutMix**: Cut and paste image regions (alpha=1.0)
- **Label Smoothing**: Smooth one-hot labels (epsilon=0.1)

## DataLoader Configuration

### Recommended Settings
```yaml
batch_size: 256          # Per GPU
num_workers: 8           # Per GPU (adjust based on CPU cores)
pin_memory: true         # Faster GPU transfer
persistent_workers: true # Keep workers alive between epochs
```

### Performance Considerations
- **num_workers**: Set to 4-12 per GPU depending on CPU cores
- **pin_memory**: Always True for GPU training
- **persistent_workers**: Reduces worker initialization overhead
- **prefetch_factor**: Default is 2, can increase for faster storage

## Testing the Data Module

### Quick Verification
```python
from data.datamodule import ImageNetDataModule

# Create data module
dm = ImageNetDataModule(
    data_root="/data2/imagenet",
    batch_size=256,
    num_workers=8
)

# Setup
dm.setup("fit")

# Check dataset sizes
print(f"Train samples: {len(dm.train_dataset)}")
print(f"Val samples: {len(dm.val_dataset)}")

# Test dataloader
train_loader = dm.train_dataloader()
batch = next(iter(train_loader))
images, labels = batch
print(f"Batch shape: {images.shape}")
print(f"Labels shape: {labels.shape}")
```

## Known Issues & Solutions

### Issue: Slow data loading
**Solution**: Increase `num_workers`, enable `persistent_workers`, or use faster storage

### Issue: Out of memory during data loading
**Solution**: Reduce `num_workers` or `batch_size`

### Issue: Workers timing out
**Solution**: Reduce `num_workers` or increase worker timeout in DataLoader

## Next Steps
1. ✅ Verify data structure and accessibility
2. ✅ Test data module with small batch
3. ⏳ Integrate with training loop
4. ⏳ Benchmark data loading throughput
5. ⏳ Optimize for multi-GPU setup
