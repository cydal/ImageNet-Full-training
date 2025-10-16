#!/bin/bash
# Mount AWS FSx filesystem at /fsx
# Requires FSx DNS name and proper IAM permissions

set -e

# Configuration
FSX_DNS_NAME="${FSX_DNS_NAME:-fs-0123456789abcdef0.fsx.us-east-1.amazonaws.com}"
MOUNT_POINT="/fsx"
MOUNT_OPTIONS="defaults,_netdev,flock,user_xattr,noatime,noauto"

echo "=========================================="
echo "Mounting FSx filesystem"
echo "=========================================="

# Check if FSx DNS name is set
if [[ "$FSX_DNS_NAME" == "fs-0123456789abcdef0.fsx.us-east-1.amazonaws.com" ]]; then
    echo "Error: Please set FSX_DNS_NAME environment variable"
    echo "Example: export FSX_DNS_NAME=fs-xxxxx.fsx.us-east-1.amazonaws.com"
    exit 1
fi

# Install Lustre client if not already installed
if ! command -v mount.lustre &> /dev/null; then
    echo "Installing Lustre client..."
    sudo apt-get update
    sudo apt-get install -y lustre-client-modules-$(uname -r) lustre-utils
fi

# Create mount point if it doesn't exist
if [ ! -d "$MOUNT_POINT" ]; then
    echo "Creating mount point: $MOUNT_POINT"
    sudo mkdir -p "$MOUNT_POINT"
fi

# Check if already mounted
if mountpoint -q "$MOUNT_POINT"; then
    echo "FSx is already mounted at $MOUNT_POINT"
    df -h "$MOUNT_POINT"
    exit 0
fi

# Extract filesystem ID from DNS name (e.g., fs-0c312f22a38ec497f)
FSX_ID=$(echo "$FSX_DNS_NAME" | cut -d'.' -f1)
echo "Filesystem ID: $FSX_ID"

# FSx mount name - can be overridden with FSX_MOUNT_NAME env var
# Default is typically the short filesystem ID without the full name
FSX_MOUNT_NAME="${FSX_MOUNT_NAME:-fsx}"
echo "Using mount name: $FSX_MOUNT_NAME"

# Mount FSx
echo "Mounting FSx at $MOUNT_POINT..."
sudo mount -t lustre -o "$MOUNT_OPTIONS" "${FSX_DNS_NAME}@tcp:/${FSX_MOUNT_NAME}" "$MOUNT_POINT"

# Verify mount
if mountpoint -q "$MOUNT_POINT"; then
    echo "FSx successfully mounted!"
    df -h "$MOUNT_POINT"
else
    echo "Error: Failed to mount FSx"
    exit 1
fi

echo "=========================================="
echo "FSx mount complete!"
echo "=========================================="
