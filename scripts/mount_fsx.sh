#!/bin/bash
# Mount AWS FSx filesystem at /fsx
# Requires FSx DNS name and proper IAM permissions

set -e

# Configuration - override with environment variables
FSX_DNS_NAME="${FSX_DNS_NAME:-}"
FSX_MOUNT_NAME="${FSX_MOUNT_NAME:-}"
MOUNT_POINT="${MOUNT_POINT:-/fsx}"
MOUNT_OPTIONS="${MOUNT_OPTIONS:-relatime,flock}"

echo "=========================================="
echo "Mounting FSx filesystem"
echo "=========================================="

# Check if FSx DNS name is set
if [[ -z "$FSX_DNS_NAME" ]]; then
    echo "Error: FSX_DNS_NAME environment variable is required"
    echo ""
    echo "Usage:"
    echo "  export FSX_DNS_NAME=fs-xxxxx.fsx.us-east-1.amazonaws.com"
    echo "  export FSX_MOUNT_NAME=your_mount_name  # Optional, will try to detect"
    echo "  $0"
    echo ""
    echo "Example:"
    echo "  export FSX_DNS_NAME=fs-0aba0b7beacfbc7bc.fsx.us-east-1.amazonaws.com"
    echo "  export FSX_MOUNT_NAME=uechfbuv"
    echo "  ./scripts/mount_fsx.sh"
    exit 1
fi

# Check if mount.lustre exists
if ! command -v mount.lustre &> /dev/null; then
    echo "Error: mount.lustre not found!"
    echo "Installing Lustre client utilities..."
    
    # Add FSx repository if not present
    if [ ! -f /etc/apt/sources.list.d/fsxlustreclientrepo.list ]; then
        echo "Adding FSx Lustre repository..."
        wget -O - https://fsx-lustre-client-repo-public-keys.s3.amazonaws.com/fsx-ubuntu-public-key.asc | \
            gpg --dearmor | sudo tee /usr/share/keyrings/fsx-ubuntu-public-key.gpg >/dev/null
        
        # Try noble (24.04) first, fall back to jammy (22.04) if needed
        UBUNTU_CODENAME=$(lsb_release -cs)
        sudo bash -c "echo 'deb [signed-by=/usr/share/keyrings/fsx-ubuntu-public-key.gpg] https://fsx-lustre-client-repo.s3.amazonaws.com/ubuntu $UBUNTU_CODENAME main' > /etc/apt/sources.list.d/fsxlustreclientrepo.list"
    fi
    
    sudo apt-get update
    sudo apt-get install -y lustre-utils
fi

# Check if Lustre kernel modules are loaded
if ! grep -q lustre /proc/filesystems 2>/dev/null; then
    echo "Warning: Lustre not found in /proc/filesystems"
    echo "Checking for Lustre kernel modules..."
    
    CURRENT_KERNEL=$(uname -r)
    echo "Current kernel: $CURRENT_KERNEL"
    
    # Try to install modules for current kernel
    if sudo apt-cache show lustre-client-modules-$CURRENT_KERNEL &>/dev/null; then
        echo "Installing Lustre modules for $CURRENT_KERNEL..."
        sudo apt-get install -y lustre-client-modules-$CURRENT_KERNEL
        
        echo "Loading Lustre modules..."
        sudo modprobe lustre || true
    else
        echo ""
        echo "ERROR: No Lustre modules available for kernel $CURRENT_KERNEL"
        echo ""
        echo "Available Lustre modules:"
        sudo apt-cache search lustre-client-modules | head -10
        echo ""
        echo "Your kernel is too new. You need to either:"
        echo "  1. Downgrade to a supported kernel (recommended)"
        echo "  2. Wait for AWS to release modules for kernel $CURRENT_KERNEL"
        echo ""
        echo "To see available kernels and install one:"
        echo "  sudo apt-cache search lustre-client-modules"
        echo ""
        echo "Example to install a compatible kernel:"
        LATEST_MODULE=$(sudo apt-cache search lustre-client-modules | grep -oP 'lustre-client-modules-\K[0-9]+\.[0-9]+\.[0-9]+-[0-9]+-aws' | sort -V | tail -1)
        if [ -n "$LATEST_MODULE" ]; then
            echo "  sudo apt-get install -y linux-image-$LATEST_MODULE linux-headers-$LATEST_MODULE"
            echo "  sudo sed -i 's/GRUB_DEFAULT=.*/GRUB_DEFAULT=\"Advanced options for Ubuntu>Ubuntu, with Linux $LATEST_MODULE\"/' /etc/default/grub"
            echo "  sudo update-grub && sudo reboot"
        fi
        exit 1
    fi
    
    # Verify modules loaded
    if ! grep -q lustre /proc/filesystems 2>/dev/null; then
        echo "ERROR: Lustre modules failed to load"
        echo "Check: lsmod | grep lustre"
        echo "Check: dmesg | tail -50"
        exit 1
    fi
fi

echo "✓ Lustre client is ready"

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

# Extract filesystem ID from DNS name (e.g., fs-0aba0b7beacfbc7bc)
FSX_ID=$(echo "$FSX_DNS_NAME" | cut -d'.' -f1)
echo "Filesystem ID: $FSX_ID"

# Determine FSx mount name
if [[ -z "$FSX_MOUNT_NAME" ]]; then
    echo "FSX_MOUNT_NAME not set, attempting to detect..."
    
    # Try to get mount name from FSx API if AWS CLI is available
    if command -v aws &> /dev/null; then
        echo "Querying FSx API for mount name..."
        FSX_MOUNT_NAME=$(aws fsx describe-file-systems \
            --file-system-ids "$FSX_ID" \
            --query 'FileSystems[0].LustreConfiguration.MountName' \
            --output text 2>/dev/null || echo "")
        
        if [[ -n "$FSX_MOUNT_NAME" && "$FSX_MOUNT_NAME" != "None" ]]; then
            echo "✓ Detected mount name: $FSX_MOUNT_NAME"
        else
            echo "Warning: Could not detect mount name from API"
            FSX_MOUNT_NAME="fsx"
            echo "Using default: $FSX_MOUNT_NAME (this may fail)"
        fi
    else
        echo "Warning: AWS CLI not available, using default mount name 'fsx'"
        FSX_MOUNT_NAME="fsx"
    fi
else
    echo "Using provided mount name: $FSX_MOUNT_NAME"
fi

# Mount FSx
echo "Mounting FSx at $MOUNT_POINT..."
sudo mount -t lustre -o "$MOUNT_OPTIONS" "${FSX_DNS_NAME}@tcp:/${FSX_MOUNT_NAME}" "$MOUNT_POINT"

# Verify mount
if mountpoint -q "$MOUNT_POINT"; then
    echo "✓ FSx successfully mounted!"
    df -h "$MOUNT_POINT"
    
    echo ""
    echo "Checking for data repository associations..."
    if command -v aws &> /dev/null; then
        DRA_INFO=$(aws fsx describe-data-repository-associations \
            --filters Name=file-system-id,Values="$FSX_ID" \
            --query 'Associations[*].{Path:FileSystemPath,S3:DataRepositoryPath,AutoImport:S3.AutoImportPolicy.Events}' \
            --output table 2>/dev/null || echo "")
        
        if [[ -n "$DRA_INFO" ]]; then
            echo "$DRA_INFO"
            echo ""
            echo "Note: If you don't see expected directories, you may need to run:"
            echo "  aws fsx create-data-repository-task --type IMPORT_METADATA_FROM_REPOSITORY --file-system-id $FSX_ID"
        fi
    fi
else
    echo "Error: Failed to mount FSx"
    exit 1
fi

echo ""
echo "=========================================="
echo "FSx mount complete!"
echo "=========================================="
echo "Mount point: $MOUNT_POINT"
echo "Filesystem: $FSX_DNS_NAME"
echo "Mount name: $FSX_MOUNT_NAME"
