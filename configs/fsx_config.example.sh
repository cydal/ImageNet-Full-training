#!/bin/bash
# FSx Configuration Example
# Copy this file to fsx_config.sh and customize for your setup
# Usage: 
#   cp configs/fsx_config.example.sh configs/fsx_config.sh
#   # Edit fsx_config.sh with your actual values
#   source configs/fsx_config.sh && ./scripts/mount_fsx.sh

# Required: Your FSx filesystem DNS name
# Get this from: AWS Console > FSx > Your filesystem > Network & security
export FSX_DNS_NAME="fs-XXXXXXXXXXXXXXXX.fsx.REGION.amazonaws.com"

# Optional: FSx mount name (auto-detected if not set)
# Get this from: AWS Console > FSx > Your filesystem > Network & security > Mount name
# Or let the script auto-detect it via AWS API
# export FSX_MOUNT_NAME="your_mount_name"

# Optional: Local mount point (default: /fsx)
# export MOUNT_POINT="/fsx"

# Optional: Lustre mount options (default: relatime,flock)
# export MOUNT_OPTIONS="relatime,flock"

# Optional: For metadata import script
# This is the filesystem ID (the fs-XXXX part from FSX_DNS_NAME)
export FSX_ID="fs-XXXXXXXXXXXXXXXX"

# Data paths (adjust based on your DRA configuration)
# Check your DRA FileSystemPath to determine the correct paths
export TRAIN_DATA_PATH="/fsx/YOUR_DRA_PATH/train"
export VAL_DATA_PATH="/fsx/YOUR_DRA_PATH/val"
