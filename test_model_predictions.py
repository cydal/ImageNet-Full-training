#!/usr/bin/env python3
"""
Test model predictions to see what's actually happening.
"""
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from pathlib import Path
from models.resnet50 import ResNet50Module

# Load the best checkpoint from LR=0.1 run
checkpoint_dir = Path("checkpoints")
checkpoints = list(checkpoint_dir.glob("**/*.ckpt"))
checkpoints = [c for c in checkpoints if "resnet50-epoch" in str(c)]
if not checkpoints:
    print("No checkpoints found!")
    exit(1)

latest_ckpt = sorted(checkpoints, key=lambda x: x.stat().st_mtime)[-1]
print(f"Loading checkpoint: {latest_ckpt}")

# Load model
model = ResNet50Module.load_from_checkpoint(
    latest_ckpt,
    num_classes=100
)
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

print(f"\nVal dataset: {len(val_dataset)} samples, {len(val_dataset.classes)} classes")

# Test on first batch
batch_size = 32
correct_top1 = 0
correct_top5 = 0
total = 0

print("\nTesting first 100 samples...")
print("="*70)

with torch.no_grad():
    for i in range(min(100, len(val_dataset))):
        image, label = val_dataset[i]
        image = image.unsqueeze(0).cuda()
        
        # Get prediction
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        
        # Top-1 and Top-5
        _, pred_top5 = output.topk(5, dim=1)
        pred_top1 = pred_top5[0, 0].item()
        pred_top5 = pred_top5[0].cpu().numpy()
        
        correct_top1 += (pred_top1 == label)
        correct_top5 += (label in pred_top5)
        total += 1
        
        # Print first 10 predictions
        if i < 10:
            max_prob = probabilities.max().item()
            print(f"Sample {i}: true={label}, pred={pred_top1}, "
                  f"match={'✓' if pred_top1==label else '✗'}, "
                  f"max_prob={max_prob:.4f}")
            print(f"  Top-5 preds: {pred_top5}")

print("="*70)
print(f"\nResults on {total} samples:")
print(f"Top-1 Accuracy: {100.0 * correct_top1 / total:.2f}%")
print(f"Top-5 Accuracy: {100.0 * correct_top5 / total:.2f}%")

# Check if model is outputting reasonable probabilities
print("\n" + "="*70)
print("PROBABILITY DISTRIBUTION CHECK")
print("="*70)

with torch.no_grad():
    image, label = val_dataset[0]
    image = image.unsqueeze(0).cuda()
    output = model(image)
    probs = F.softmax(output, dim=1)[0]
    
    print(f"\nSample 0 (true label={label}):")
    print(f"  Min prob: {probs.min().item():.6f}")
    print(f"  Max prob: {probs.max().item():.6f}")
    print(f"  Mean prob: {probs.mean().item():.6f}")
    print(f"  Std prob: {probs.std().item():.6f}")
    print(f"  Entropy: {-(probs * torch.log(probs + 1e-10)).sum().item():.4f}")
    
    # Show top 10 predictions
    top10_probs, top10_idx = probs.topk(10)
    print(f"\n  Top 10 predictions:")
    for i, (prob, idx) in enumerate(zip(top10_probs, top10_idx)):
        marker = "✓" if idx.item() == label else " "
        print(f"    {i+1}. Class {idx.item():3d}: {prob.item():.4f} {marker}")
