# FSx Setup Workflow

## When You Create a New FSx Filesystem

### Step 1: Get FSx Information

```bash
# List available FSx filesystems
aws fsx describe-file-systems \
    --query 'FileSystems[*].{ID:FileSystemId,DNS:DNSName,MountName:LustreConfiguration.MountName,Status:Lifecycle}' \
    --output table

# Example output:
# ID: fs-02386cb09beeabb62
# DNS: fs-02386cb09beeabb62.fsx.us-east-1.amazonaws.com
# MountName: ujwu7buv
```

### Step 2: Create Your Config File

```bash
cd /home/ubuntu/ImageNet-Full-training

# Copy the example config
cp configs/fsx_config.example.sh configs/fsx_config.sh

# Edit with your actual values
nano configs/fsx_config.sh
```

**Fill in these values:**
```bash
export FSX_DNS_NAME="fs-XXXXXXXXXXXXXXXX.fsx.REGION.amazonaws.com"
export FSX_MOUNT_NAME="your_mount_name"
export FSX_ID="fs-XXXXXXXXXXXXXXXX"
export DRA_BASE_PATH="/ns1"  # Or your DRA path
```

### Step 3: Mount FSx with Auto Lazy-Loading

```bash
# Source config and mount
source configs/fsx_config.sh && ./scripts/mount_fsx.sh
```

**What this does:**
1. ✓ Mounts FSx at `/fsx`
2. ✓ Automatically triggers lazy-loading for `train/` and `val/`
3. ✓ Verifies class counts (1000 each)
4. ✓ Shows any missing directories

**Expected output:**
```
==========================================
Mounting FSx filesystem
==========================================
✓ Lustre client is ready
Filesystem ID: fs-02386cb09beeabb62
Using provided mount name: ujwu7buv
Mounting FSx at /fsx...
✓ FSx successfully mounted!

==========================================
Checking and lazy-loading directories...
==========================================
Using directories from config: 2 paths

Triggering lazy-load for all directories...
Checking: /fsx/ns1/train ... ✓ (1000 items)
Checking: /fsx/ns1/val ... ✓ (1000 items)

Lazy-loaded 2 out of 2 directories

Verifying class counts...
✓ Train: 1000 classes (expected 1000)
✓ Val: 1000 classes (expected 1000)

==========================================
FSx mount complete!
==========================================
```

## Current FSx Details (for reference)

**Your current filesystem:**
- **ID:** `fs-02386cb09beeabb62`
- **DNS:** `fs-02386cb09beeabb62.fsx.us-east-1.amazonaws.com`
- **Mount name:** `ujwu7buv`
- **DRA path:** `/ns1`
- **S3 bucket:** `s3://sij-imagenet-train/imagenet/`

## Quick Commands

```bash
# Mount FSx (after creating fsx_config.sh)
source configs/fsx_config.sh && ./scripts/mount_fsx.sh

# Verify data
ls -1 /fsx/ns1/train | wc -l  # Should show 1000
ls -1 /fsx/ns1/val | wc -l    # Should show 1000

# Unmount
sudo umount /fsx

# Check mount status
mountpoint /fsx && echo "Mounted" || echo "Not mounted"
```

## Troubleshooting

### val folder not visible after mount

**This is normal!** FSx uses lazy-loading. The mount script now automatically triggers it, but if you need to manually trigger:

```bash
# Access the directory to trigger lazy-load
ls /fsx/ns1/val

# Now it will be visible
ls -1 /fsx/ns1/val | wc -l
```

### Need to import metadata manually

```bash
# If directories still missing after lazy-load attempt
export FSX_ID="fs-XXXXXXXXXXXXXXXX"
./scripts/import_fsx_metadata.sh
```

## Files Overview

| File | Purpose | Committed to Git? |
|------|---------|-------------------|
| `configs/fsx_config.example.sh` | Template with placeholders | ✓ Yes |
| `configs/fsx_config.sh` | Your actual FSx values | ✗ No (gitignored) |
| `scripts/mount_fsx.sh` | Mount script with lazy-loading | ✓ Yes |
| `scripts/import_fsx_metadata.sh` | Manual metadata import | ✓ Yes |

## Why This Works

1. **`fsx_config.sh` is gitignored** - Your actual FSx IDs never get committed
2. **`mount_fsx.sh` reads from environment** - No hardcoded values
3. **Automatic lazy-loading** - Script accesses directories to trigger FSx lazy-load
4. **Verification** - Confirms expected class counts after mount

## Next Time You Spin Up FSx

1. Create new FSx filesystem in AWS
2. Get the new FSx ID, DNS, and mount name
3. Edit `configs/fsx_config.sh` with new values
4. Run: `source configs/fsx_config.sh && ./scripts/mount_fsx.sh`
5. Done! Both `train/` and `val/` will be visible and verified
