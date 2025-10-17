#!/bin/bash
# FSx Configuration
# Copy this file to fsx_config.sh and customize for your setup
# Usage: 
#   cp configs/fsx_config.example.sh configs/fsx_config.sh
#   # Edit fsx_config.sh with your actual values
#   source configs/fsx_config.sh && ./scripts/mount_fsx.sh

# ============================================================================
# REQUIRED: FSx Filesystem Configuration
# ============================================================================
# Get these from: AWS Console > FSx > Your filesystem > Network & security
# Or run: aws fsx describe-file-systems --query 'FileSystems[*].{ID:FileSystemId,DNS:DNSName,MountName:LustreConfiguration.MountName}'

# FSx filesystem DNS name (REQUIRED)
export FSX_DNS_NAME="fs-XXXXXXXXXXXXXXXX.fsx.REGION.amazonaws.com"

# FSx mount name (REQUIRED - get from AWS console or API)
export FSX_MOUNT_NAME="your_mount_name"

# FSx filesystem ID (REQUIRED - for metadata operations)
export FSX_ID="fs-XXXXXXXXXXXXXXXX"

# ============================================================================
# OPTIONAL: Mount Configuration
# ============================================================================

# Local mount point (default: /fsx)
export MOUNT_POINT="/fsx"

# Lustre mount options (default: relatime,flock)
export MOUNT_OPTIONS="relatime,flock"

# ============================================================================
# DATA PATHS: Expected directories to verify/trigger lazy-load
# ============================================================================
# These paths are relative to MOUNT_POINT
# The mount script will check these and trigger lazy-loading if needed

# DRA base path (where your S3 data is linked)
export DRA_BASE_PATH="/ns1"

# Expected subdirectories (will be checked and lazy-loaded)
export EXPECTED_DIRS=(
    "${DRA_BASE_PATH}/train"
    "${DRA_BASE_PATH}/val"
)

# ============================================================================
# VERIFICATION: Expected class counts
# ============================================================================
# Used to verify data integrity after mount

export EXPECTED_TRAIN_CLASSES=1000
export EXPECTED_VAL_CLASSES=1000
