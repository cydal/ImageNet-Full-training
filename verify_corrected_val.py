#!/usr/bin/env python3
"""
Quick verification that the corrected validation data is properly labeled.
Uses a pretrained 1000-class ImageNet model to check if images match their folder labels.
"""
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from pathlib import Path
import json

# Load ImageNet class index (synset to class ID mapping)
# This maps WordNet IDs (like n01498041) to ImageNet 1000-class indices
IMAGENET_CLASS_INDEX_URL = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"

print("Loading ImageNet class index...")
import urllib.request
with urllib.request.urlopen(IMAGENET_CLASS_INDEX_URL) as url:
    class_idx = json.loads(url.read().decode())

# Create synset to index mapping
synset_to_idx = {v[0]: int(k) for k, v in class_idx.items()}

print(f"Loaded {len(synset_to_idx)} ImageNet classes")
print()

# Load pretrained ResNet50
print("Loading pretrained ResNet50...")
model = models.resnet50(pretrained=True)
model.eval()
model = model.cuda()

# Load validation dataset
data_root = Path("/mnt/nvme_data/imagenet_subset")
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder(root=data_root / "val", transform=val_transform)

print(f"Val dataset: {len(val_dataset)} samples, {len(val_dataset.classes)} classes")
print()

# Test a sample from each class
print("="*70)
print("VERIFICATION: Testing one sample per class")
print("="*70)
print()

correct = 0
total = 0
mismatches = []

with torch.no_grad():
    for class_idx_local, class_name in enumerate(val_dataset.classes):
        # Get expected ImageNet class index
        if class_name not in synset_to_idx:
            print(f"WARNING: {class_name} not in ImageNet class index")
            continue
        
        expected_imagenet_idx = synset_to_idx[class_name]
        
        # Find first sample of this class
        sample_idx = None
        for i, (_, label) in enumerate(val_dataset.samples):
            if label == class_idx_local:
                sample_idx = i
                break
        
        if sample_idx is None:
            print(f"WARNING: No samples found for class {class_name}")
            continue
        
        # Get prediction
        image, _ = val_dataset[sample_idx]
        image = image.unsqueeze(0).cuda()
        
        output = model(image)
        probs = F.softmax(output, dim=1)
        top5_probs, top5_idx = probs.topk(5)
        
        predicted_idx = top5_idx[0, 0].item()
        predicted_prob = top5_probs[0, 0].item()
        
        # Check if prediction matches expected
        match = predicted_idx == expected_imagenet_idx
        if match:
            correct += 1
            marker = "✓"
        else:
            marker = "✗"
            mismatches.append({
                'class': class_name,
                'expected': expected_imagenet_idx,
                'predicted': predicted_idx,
                'prob': predicted_prob,
                'top5': top5_idx[0].cpu().tolist()
            })
        
        total += 1
        
        # Print first 20 and last 20
        if total <= 20 or total > len(val_dataset.classes) - 20:
            print(f"{marker} Class {class_idx_local:2d} ({class_name}): "
                  f"expected={expected_imagenet_idx:3d}, "
                  f"predicted={predicted_idx:3d} ({predicted_prob:.3f})")

print()
print("="*70)
print("RESULTS")
print("="*70)
print(f"Total classes tested: {total}")
print(f"Correct predictions: {correct}")
print(f"Accuracy: {100.0 * correct / total:.1f}%")
print()

if mismatches:
    print(f"Found {len(mismatches)} mismatches:")
    print()
    for i, mm in enumerate(mismatches[:10]):
        print(f"{i+1}. {mm['class']}: expected {mm['expected']}, "
              f"predicted {mm['predicted']} ({mm['prob']:.3f})")
        print(f"   Top-5: {mm['top5']}")
    
    if len(mismatches) > 10:
        print(f"   ... and {len(mismatches) - 10} more")
    print()

# Interpretation
print("="*70)
print("INTERPRETATION")
print("="*70)
if correct / total > 0.8:
    print("✅ VALIDATION DATA LOOKS CORRECT!")
    print("   Most images match their folder labels.")
    print("   Safe to proceed with training.")
elif correct / total > 0.5:
    print("⚠️  VALIDATION DATA PARTIALLY CORRECT")
    print("   Some images match their labels, but there are issues.")
    print("   Review the mismatches above.")
else:
    print("❌ VALIDATION DATA STILL INCORRECT!")
    print("   Most images don't match their folder labels.")
    print("   DO NOT proceed with training until this is fixed.")

print("="*70)
