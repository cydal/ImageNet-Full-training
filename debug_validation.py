#!/usr/bin/env python3
"""
Debug validation to find why val accuracy is stuck at 1%.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from models.resnet50 import ResNet50Module
from tqdm import tqdm

# Load latest pretrained checkpoint
checkpoint_dir = Path("checkpoints")
checkpoints = list(checkpoint_dir.glob("**/*.ckpt"))
checkpoints = [c for c in checkpoints if "resnet50-epoch" in str(c)]
latest_ckpt = sorted(checkpoints, key=lambda x: x.stat().st_mtime)[-1]
print(f"Loading checkpoint: {latest_ckpt}\n")

# Load model
model = ResNet50Module.load_from_checkpoint(latest_ckpt, num_classes=100)
model.eval()
model = model.cuda()

# Load val dataset
data_root = Path("/mnt/nvme_data/imagenet_subset")
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder(root=data_root / "val", transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)

print(f"Val dataset: {len(val_dataset)} samples, {len(val_dataset.classes)} classes")
print(f"Val batches: {len(val_loader)}\n")

# Manual validation
correct_top1 = 0
correct_top5 = 0
total = 0
class_correct = torch.zeros(100)
class_total = torch.zeros(100)
all_preds = []
all_labels = []

print("Running validation...")
with torch.no_grad():
    for images, labels in tqdm(val_loader):
        images = images.cuda()
        labels_cuda = labels.cuda()
        
        # Forward pass
        outputs = model(images)
        
        # Top-1 and Top-5
        _, pred_top5 = outputs.topk(5, dim=1)
        pred_top1 = pred_top5[:, 0]
        
        # Accuracy
        correct_top1 += (pred_top1 == labels_cuda).sum().item()
        for i in range(5):
            correct_top5 += (pred_top5[:, i] == labels_cuda).sum().item()
        
        total += labels.size(0)
        
        # Per-class accuracy
        for i in range(labels.size(0)):
            label = labels[i].item()
            class_total[label] += 1
            if pred_top1[i].item() == label:
                class_correct[label] += 1
        
        # Store for analysis
        all_preds.extend(pred_top1.cpu().numpy())
        all_labels.extend(labels.numpy())

# Results
top1_acc = 100.0 * correct_top1 / total
top5_acc = 100.0 * correct_top5 / total

print("\n" + "="*70)
print("VALIDATION RESULTS")
print("="*70)
print(f"Total samples: {total}")
print(f"Top-1 Accuracy: {top1_acc:.2f}%")
print(f"Top-5 Accuracy: {top5_acc:.2f}%")

# Per-class accuracy
print("\n" + "="*70)
print("PER-CLASS ACCURACY (first 20 classes)")
print("="*70)
for i in range(min(20, 100)):
    if class_total[i] > 0:
        acc = 100.0 * class_correct[i] / class_total[i]
        print(f"Class {i:2d}: {acc:5.1f}% ({int(class_correct[i])}/{int(class_total[i])})")

# Prediction distribution
import numpy as np
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

print("\n" + "="*70)
print("PREDICTION DISTRIBUTION")
print("="*70)
unique_preds, counts = np.unique(all_preds, return_counts=True)
top_predicted = sorted(zip(unique_preds, counts), key=lambda x: x[1], reverse=True)[:10]
print("Top 10 most predicted classes:")
for cls, count in top_predicted:
    print(f"  Class {cls:2d}: {count:4d} predictions ({100.0*count/total:.1f}%)")

print("\n" + "="*70)
print("LABEL DISTRIBUTION")
print("="*70)
unique_labels, counts = np.unique(all_labels, return_counts=True)
print(f"Unique labels in val set: {len(unique_labels)}")
print(f"Samples per class: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")

# Check if there's a mismatch
print("\n" + "="*70)
print("SANITY CHECKS")
print("="*70)
print(f"All labels in range [0, 99]: {all_labels.min() >= 0 and all_labels.max() <= 99}")
print(f"All preds in range [0, 99]: {all_preds.min() >= 0 and all_preds.max() <= 99}")
print(f"Number of unique labels: {len(unique_labels)} (should be 100)")
print(f"Number of unique preds: {len(unique_preds)}")
