#!/bin/bash
# Mount S3 bucket directly using mountpoint-s3 (AWS's official S3 mount tool)
# This is faster and more reliable than s3fs

set -e

echo "================================================================================"
echo "S3 Direct Mount Setup"
echo "================================================================================"
echo ""

# Configuration
S3_BUCKET="s3://sij-imagenet-train"
MOUNT_POINT="/mnt/s3-imagenet"
CACHE_DIR="/tmp/s3-cache"

# Load AWS credentials if not already set
if [ -z "$AWS_ACCESS_KEY_ID" ]; then
    if [ -f "$HOME/.aws/credentials" ]; then
        export AWS_ACCESS_KEY_ID=$(grep -A2 "\[default\]" "$HOME/.aws/credentials" | grep aws_access_key_id | cut -d'=' -f2 | tr -d ' ')
        export AWS_SECRET_ACCESS_KEY=$(grep -A2 "\[default\]" "$HOME/.aws/credentials" | grep aws_secret_access_key | cut -d'=' -f2 | tr -d ' ')
    fi
fi

if [ -z "$AWS_DEFAULT_REGION" ]; then
    if [ -f "$HOME/.aws/config" ]; then
        export AWS_DEFAULT_REGION=$(grep -A2 "\[default\]" "$HOME/.aws/config" | grep region | cut -d'=' -f2 | tr -d ' ')
    fi
fi

# Check if mountpoint-s3 is installed
if ! command -v mount-s3 &> /dev/null; then
    echo "Installing mountpoint-s3..."
    echo ""
    
    # Download and install mountpoint-s3
    wget https://s3.amazonaws.com/mountpoint-s3-release/latest/x86_64/mount-s3.deb
    sudo apt-get install -y ./mount-s3.deb
    rm mount-s3.deb
    
    echo "✓ mountpoint-s3 installed"
    echo ""
fi

# Create mount point and cache directory
echo "Creating directories..."
sudo mkdir -p "$MOUNT_POINT"
mkdir -p "$CACHE_DIR"

# Check if already mounted
if mountpoint -q "$MOUNT_POINT"; then
    echo "⚠️  $MOUNT_POINT is already mounted"
    echo ""
    echo "To unmount: sudo umount $MOUNT_POINT"
    exit 0
fi

# Mount S3 bucket
echo "Mounting $S3_BUCKET to $MOUNT_POINT..."
echo ""

# Mount with optimizations for ML workloads
# Pass AWS credentials to sudo environment
sudo -E AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
     AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
     AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}" \
     mount-s3 \
    --cache "$CACHE_DIR" \
    --max-threads 32 \
    --part-size 8388608 \
    --allow-other \
    --read-only \
    sij-imagenet-train "$MOUNT_POINT"

echo "✓ S3 bucket mounted successfully!"
echo ""

# Verify mount
echo "Verifying mount..."
if [ -d "$MOUNT_POINT/imagenet/train" ] && [ -d "$MOUNT_POINT/imagenet/val" ]; then
    echo "✓ Data directories found:"
    echo "  - $MOUNT_POINT/imagenet/train"
    echo "  - $MOUNT_POINT/imagenet/val"
    echo ""
    
    # Count classes (this may take a moment on first access)
    echo "  Counting classes (first access may be slow)..."
    train_classes=$(ls -1 "$MOUNT_POINT/imagenet/train" 2>/dev/null | wc -l)
    val_classes=$(ls -1 "$MOUNT_POINT/imagenet/val" 2>/dev/null | wc -l)
    
    echo "  Train classes: $train_classes"
    echo "  Val classes: $val_classes"
else
    echo "✗ Warning: Expected directories not found at standard locations"
    echo "  Checking bucket contents..."
    ls -la "$MOUNT_POINT/" 2>/dev/null || echo "  Could not list mount point"
fi

echo ""
echo "================================================================================"
echo "Mount complete!"
echo "================================================================================"
echo ""
echo "Data available at: $MOUNT_POINT"
echo "Cache directory: $CACHE_DIR"
echo ""
echo "To unmount:"
echo "  sudo umount $MOUNT_POINT"
echo ""
echo "Note: First access will be slow (S3 latency), subsequent reads use cache"
echo ""
