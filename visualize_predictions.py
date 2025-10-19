#!/usr/bin/env python3
"""
Visualize a few validation samples with predictions to debug the issue.
"""
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from pathlib import Path
from models.resnet50 import ResNet50Module
from PIL import Image

# Load latest checkpoint
checkpoint_dir = Path("checkpoints")
checkpoints = list(checkpoint_dir.glob("**/*.ckpt"))
checkpoints = [c for c in checkpoints if "resnet50-epoch" in str(c)]
latest_ckpt = sorted(checkpoints, key=lambda x: x.stat().st_mtime)[-1]
print(f"Loading checkpoint: {latest_ckpt}\n")

model = ResNet50Module.load_from_checkpoint(latest_ckpt, num_classes=100)
model.eval()
model = model.cuda()

# Load datasets
data_root = Path("/mnt/nvme_data/imagenet_subset")

# Load WITHOUT transforms first to see raw images
train_dataset_raw = datasets.ImageFolder(root=data_root / "train")
val_dataset_raw = datasets.ImageFolder(root=data_root / "val")

# Load WITH transforms for model
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_dataset = datasets.ImageFolder(root=data_root / "val", transform=val_transform)

print("="*70)
print("CHECKING CLASS NAMES AND LABELS")
print("="*70)

print(f"\nTrain classes (first 10): {train_dataset_raw.classes[:10]}")
print(f"Val classes (first 10): {val_dataset_raw.classes[:10]}")

print(f"\nTrain class_to_idx (first 10):")
for i, (cls, idx) in enumerate(list(train_dataset_raw.class_to_idx.items())[:10]):
    print(f"  {cls}: {idx}")

print(f"\nVal class_to_idx (first 10):")
for i, (cls, idx) in enumerate(list(val_dataset_raw.class_to_idx.items())[:10]):
    print(f"  {cls}: {idx}")

# Get a few samples from class 0
print("\n" + "="*70)
print("TESTING CLASS 0 SAMPLES")
print("="*70)

class_0_indices = [i for i, (_, label) in enumerate(val_dataset_raw.samples) if label == 0]
print(f"\nFound {len(class_0_indices)} samples for class 0 in val set")
print(f"Class 0 name: {val_dataset_raw.classes[0]}")

# Test first 5 samples from class 0
print("\nPredictions for class 0 samples:")
with torch.no_grad():
    for i, idx in enumerate(class_0_indices[:5]):
        img, label = val_dataset[idx]
        img_tensor = img.unsqueeze(0).cuda()
        
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)[0]
        top5_probs, top5_idx = probs.topk(5)
        
        print(f"\n  Sample {i} (index {idx}):")
        print(f"    True label: {label} ({val_dataset_raw.classes[label]})")
        print(f"    Top-5 predictions:")
        for j, (prob, pred_idx) in enumerate(zip(top5_probs, top5_idx)):
            pred_class = val_dataset_raw.classes[pred_idx.item()]
            marker = "✓" if pred_idx.item() == label else " "
            print(f"      {j+1}. Class {pred_idx.item():2d} ({pred_class}): {prob.item():.4f} {marker}")

# Now check if maybe the model was trained on DIFFERENT classes
print("\n" + "="*70)
print("HYPOTHESIS: Model trained on different 100 classes?")
print("="*70)

# The pretrained model was trained on 1000 ImageNet classes
# When we load it and replace the final layer with 100 classes,
# we lose the original predictions

# Let's check what happens if we test with the ORIGINAL 1000-class head
print("\nLoading model with ORIGINAL 1000-class head...")
model_1000 = ResNet50Module(num_classes=1000, pretrained=True)
model_1000.eval()
model_1000 = model_1000.cuda()

print("\nPredictions with 1000-class pretrained model:")
with torch.no_grad():
    for i, idx in enumerate(class_0_indices[:3]):
        img, label = val_dataset[idx]
        img_tensor = img.unsqueeze(0).cuda()
        
        output = model_1000(img_tensor)
        probs = F.softmax(output, dim=1)[0]
        top5_probs, top5_idx = probs.topk(5)
        
        print(f"\n  Sample {i}:")
        print(f"    True class: {val_dataset_raw.classes[label]}")
        print(f"    Top-5 predictions (from 1000 classes):")
        for j, (prob, pred_idx) in enumerate(zip(top5_probs, top5_idx)):
            print(f"      {j+1}. Class {pred_idx.item():3d}: {prob.item():.4f}")
