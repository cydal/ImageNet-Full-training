#!/usr/bin/env python3
"""
Sanity check: Can the model overfit a tiny batch?
If this fails, there's a fundamental bug in the training setup.
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import resnet50
from pathlib import Path

# Simple training loop on one batch
def test_overfit():
    device = torch.device("cuda:0")
    
    # Load one batch
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(
        root="/opt/dlami/nvme/imagenet/train",
        transform=transform
    )
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels[:10]}")
    print(f"Label range: {labels.min().item()} - {labels.max().item()}")
    
    # Create model
    model = resnet50(weights=None, num_classes=1000).to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    
    # Try to overfit this single batch
    print("\nAttempting to overfit single batch...")
    for step in range(100):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()
        acc = 100.0 * correct / labels.size(0)
        
        if step % 10 == 0:
            print(f"Step {step:3d}: Loss={loss.item():.4f}, Acc={acc:.1f}%")
    
    print("\n" + "="*50)
    if acc > 90:
        print("✓ PASS: Model can learn (overfitted to 90%+)")
    else:
        print("✗ FAIL: Model cannot learn - fundamental bug!")
    print("="*50)

if __name__ == "__main__":
    test_overfit()
