# FSx for Lustre Setup Guide

This guide covers mounting and troubleshooting AWS FSx for Lustre filesystems for ImageNet training.

## Quick Start

### 1. Configure FSx (First Time Only)

```bash
# Copy the example config
cp configs/fsx_config.example.sh configs/fsx_config.sh

# Edit with your actual FSx details
nano configs/fsx_config.sh
# Set FSX_DNS_NAME, FSX_MOUNT_NAME, etc.
```

**Important**: `fsx_config.sh` is gitignored and contains your actual filesystem IDs. Never commit this file.

### 2. Mount FSx

```bash
# Option A: Source the config file
source configs/fsx_config.sh
./scripts/mount_fsx.sh

# Option B: Set environment variables directly
export FSX_DNS_NAME=fs-xxxxx.fsx.us-east-1.amazonaws.com
export FSX_MOUNT_NAME=your_mount_name  # Optional, will auto-detect
./scripts/mount_fsx.sh
```

The script will:
- Install Lustre client if needed (handles kernel compatibility)
- Auto-detect the FSx mount name from AWS API
- Mount the filesystem at `/fsx`
- Show Data Repository Association (DRA) information

### 3. Verify Data

```bash
ls -la /fsx/
```

If you see your expected directories (e.g., `train/`, `val/`), you're done!

## Troubleshooting

### Missing Directories After Mount

**Symptom**: FSx mounts successfully but expected directories from S3 are missing.

**Cause**: FSx metadata not imported from S3. This happens when:
- Data was uploaded to S3 after the DRA was created
- Initial metadata import didn't complete
- AutoImport is disabled

**Solution**: Run metadata import

```bash
export FSX_ID=fs-xxxxx
./scripts/import_fsx_metadata.sh
```

This will:
- Re-scan S3 and import all metadata
- Take 5-15 minutes depending on dataset size
- Show progress and completion status

### Kernel Compatibility Issues

**Symptom**: Mount fails with "No such device" or "Lustre modules not loaded"

**Cause**: Your kernel version doesn't have pre-built Lustre modules from AWS.

**Solution**: The `mount_fsx.sh` script handles this automatically by:
1. Checking if modules exist for your kernel
2. If not, showing available compatible kernels
3. Providing commands to install a compatible kernel

Manual fix:
```bash
# Find available kernels
sudo apt-cache search lustre-client-modules

# Install compatible kernel (example)
KERNEL_VERSION="6.8.0-1018-aws"
sudo apt-get install -y \
    linux-image-$KERNEL_VERSION \
    linux-headers-$KERNEL_VERSION \
    lustre-client-modules-$KERNEL_VERSION \
    lustre-utils

# Configure GRUB to use this kernel
sudo sed -i "s/GRUB_DEFAULT=.*/GRUB_DEFAULT=\"Advanced options for Ubuntu>Ubuntu, with Linux $KERNEL_VERSION\"/" /etc/default/grub
sudo update-grub
sudo reboot
```

### Getting FSx Mount Name

The mount name is auto-detected, but you can find it manually:

```bash
# Via AWS CLI
aws fsx describe-file-systems \
    --file-system-ids fs-xxxxx \
    --query 'FileSystems[0].LustreConfiguration.MountName' \
    --output text

# Via AWS Console
# FSx Console > Your filesystem > Network & security tab > Mount name
```

## Environment Variables

All scripts support these environment variables:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `FSX_DNS_NAME` | Yes | - | FSx DNS endpoint (e.g., `fs-xxxxx.fsx.region.amazonaws.com`) |
| `FSX_MOUNT_NAME` | No | Auto-detect | Lustre mount name (auto-detected from API) |
| `FSX_ID` | For import | - | Filesystem ID (e.g., `fs-xxxxx`) |
| `MOUNT_POINT` | No | `/fsx` | Local mount point |
| `MOUNT_OPTIONS` | No | `relatime,flock` | Lustre mount options |

## Data Repository Associations (DRA)

DRAs link your FSx filesystem to S3 buckets. Key concepts:

- **FileSystemPath**: Where S3 data appears in FSx (e.g., `/ns1`)
- **DataRepositoryPath**: S3 location (e.g., `s3://bucket/prefix/`)
- **AutoImport**: If enabled, new S3 objects appear automatically in FSx
- **BatchImportMetaDataOnCreate**: One-time import when DRA is created

### Enable Auto-Import

To automatically sync new S3 objects to FSx:

```bash
aws fsx update-data-repository-association \
    --association-id dra-xxxxx \
    --s3 'AutoImportPolicy={Events=[NEW,CHANGED,DELETED]}'
```

### Check DRA Status

```bash
aws fsx describe-data-repository-associations \
    --filters Name=file-system-id,Values=fs-xxxxx \
    --query 'Associations[*].{Path:FileSystemPath,S3:DataRepositoryPath,AutoImport:S3.AutoImportPolicy}' \
    --output table
```

## Multi-Instance Setup

When using FSx across multiple instances:

1. **Same region**: All instances can mount the same FSx filesystem
2. **Same VPC/subnet**: Ensure instances have network access to FSx
3. **IAM permissions**: Instances need `fsx:DescribeFileSystems` for auto-detection
4. **Kernel compatibility**: Each instance needs compatible Lustre modules

### Example: Launch Script for New Instances

```bash
#!/bin/bash
# user-data or startup script

# Set FSx configuration
export FSX_DNS_NAME=fs-0aba0b7beacfbc7bc.fsx.us-east-1.amazonaws.com

# Clone repo (or use existing)
cd /home/ubuntu/ImageNet-Full-training

# Mount FSx (handles Lustre client installation)
./scripts/mount_fsx.sh

# Verify
ls -la /fsx/ns1/train/
ls -la /fsx/ns1/val/

# Start training
./scripts/launch_single.sh
```

## Performance Tips

1. **Mount options**: Default `relatime,flock` is good for most cases
2. **Stripe settings**: For large files, consider adjusting Lustre striping
3. **Concurrent access**: FSx supports multiple readers/writers simultaneously
4. **Monitoring**: Check FSx CloudWatch metrics for throughput and IOPS

## Common Issues

### Issue: "mount.lustre: mount ... failed: No such device"
**Fix**: Install Lustre kernel modules (see Kernel Compatibility section)

### Issue: "Permission denied" when mounting
**Fix**: Use `sudo` or run script with sudo privileges

### Issue: Directories empty after mount
**Fix**: Run metadata import (see Missing Directories section)

### Issue: Mount succeeds but data is stale
**Fix**: Enable AutoImport or run manual metadata import

### Issue: Different mount name on new instance
**Fix**: Set `FSX_MOUNT_NAME` explicitly or ensure AWS CLI is configured

## References

- [AWS FSx for Lustre Documentation](https://docs.aws.amazon.com/fsx/latest/LustreGuide/)
- [Lustre Client Installation](https://docs.aws.amazon.com/fsx/latest/LustreGuide/install-lustre-client.html)
- [Data Repository Tasks](https://docs.aws.amazon.com/fsx/latest/LustreGuide/data-repository-tasks.html)
