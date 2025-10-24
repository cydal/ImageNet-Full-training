# FSx Mount Setup - Complete ✓

## What Was Fixed

### Problem
- `val` folder was not visible after mounting FSx
- FSx uses **lazy-loading** - directories don't appear until first accessed
- Each time you create a new FSx filesystem, you need to update mount details

### Solution
1. **Config file approach** - Store FSx details in `configs/fsx_config.sh` (gitignored)
2. **Automatic lazy-loading** - Mount script now triggers lazy-load for all expected directories
3. **Verification** - Confirms class counts after mount

## Files Structure

```
configs/
├── fsx_config.example.sh   # Template (committed to git)
└── fsx_config.sh           # Your actual values (gitignored)

scripts/
├── mount_fsx.sh            # Mount script with lazy-loading
└── import_fsx_metadata.sh  # Manual metadata import (if needed)
```

## The Workflow

### One-Time Setup (per FSx filesystem)

```bash
cd /home/ubuntu/ImageNet-Full-training

# Option 1: Copy and edit
cp configs/fsx_config.example.sh configs/fsx_config.sh
nano configs/fsx_config.sh  # Fill in your FSx details

# Option 2: Already created for you with current values
# configs/fsx_config.sh exists with:
#   FSX_ID: fs-02386cb09beeabb62
#   DNS: fs-02386cb09beeabb62.fsx.us-east-1.amazonaws.com
#   Mount: ujwu7buv
```

### Every Time You Mount

```bash
source configs/fsx_config.sh && ./scripts/mount_fsx.sh
```

**That's it!** The script will:
1. ✓ Mount FSx at `/fsx`
2. ✓ Auto-detect DRA paths from AWS
3. ✓ Trigger lazy-loading for `train/` and `val/`
4. ✓ Verify 1000 classes in each directory

## Verified Working

```
==========================================
Checking and lazy-loading directories...
==========================================
Auto-detected from DRA: 2 paths

Triggering lazy-load for all directories...
Checking: /fsx/ns1/train ... ✓ (1000 items)
Checking: /fsx/ns1/val ... ✓ (1000 items)

Lazy-loaded 2 out of 2 directories

Verifying class counts...
✓ Train: 1000 classes (expected 1000)
✓ Val: 1000 classes (expected 1000)
```

## When You Create a New FSx Filesystem

1. **Delete old filesystem** (to save costs)
2. **Create new FSx filesystem** in AWS console
3. **Get new details:**
   ```bash
   aws fsx describe-file-systems \
       --query 'FileSystems[*].{ID:FileSystemId,DNS:DNSName,MountName:LustreConfiguration.MountName}' \
       --output table
   ```
4. **Update `configs/fsx_config.sh`:**
   ```bash
   nano configs/fsx_config.sh
   # Update FSX_DNS_NAME, FSX_MOUNT_NAME, and FSX_ID
   ```
5. **Mount:**
   ```bash
   source configs/fsx_config.sh && ./scripts/mount_fsx.sh
   ```

## Key Design Decisions

### ✓ No Hardcoded Values
- `mount_fsx.sh` has NO hardcoded FSx IDs
- All values come from environment variables
- Safe to commit to git

### ✓ Config File is Gitignored
- `configs/fsx_config.sh` is in `.gitignore`
- Your actual FSx IDs never get committed
- Each developer/instance has their own config

### ✓ Automatic Lazy-Loading
- Script automatically accesses directories to trigger FSx lazy-load
- No manual `ls` commands needed
- Both `train/` and `val/` visible immediately after mount

### ✓ Verification Built-In
- Checks expected class counts (1000 each)
- Shows warnings if counts don't match
- Confirms data integrity

## Current Configuration

**Your `configs/fsx_config.sh` contains:**
```bash
export FSX_DNS_NAME="fs-02386cb09beeabb62.fsx.us-east-1.amazonaws.com"
export FSX_MOUNT_NAME="ujwu7buv"
export FSX_ID="fs-02386cb09beeabb62"
export DRA_BASE_PATH="/ns1"
export EXPECTED_DIRS=("/ns1/train" "/ns1/val")
export EXPECTED_TRAIN_CLASSES=1000
export EXPECTED_VAL_CLASSES=1000
```

## Quick Commands

```bash
# Mount FSx
source configs/fsx_config.sh && ./scripts/mount_fsx.sh

# Verify data
ls -1 /fsx/ns1/train | wc -l  # 1000
ls -1 /fsx/ns1/val | wc -l    # 1000

# Unmount
sudo umount /fsx

# Check status
mountpoint /fsx && echo "Mounted" || echo "Not mounted"
```

## What Happens Behind the Scenes

1. **Mount FSx** - Standard Lustre mount
2. **Query DRA** - Get linked S3 paths from AWS API
3. **Trigger Lazy-Load** - Run `ls` on each expected directory
4. **Verify Counts** - Check class counts match expectations
5. **Report Status** - Show what's available

## No More Manual Steps

**Before:**
```bash
# Mount
sudo mount -t lustre fs-XXX.fsx.us-east-1.amazonaws.com@tcp:/mount_name /fsx

# Manually trigger lazy-load
ls /fsx/ns1/train
ls /fsx/ns1/val  # Oops, forgot this one!

# Check counts
ls -1 /fsx/ns1/train | wc -l
ls -1 /fsx/ns1/val | wc -l
```

**Now:**
```bash
source configs/fsx_config.sh && ./scripts/mount_fsx.sh
# Everything done automatically!
```

## Ready for Training

Your data is now ready at:
- **Train:** `/fsx/ns1/train` (1000 classes)
- **Val:** `/fsx/ns1/val` (1000 classes)

Training configs already point to `/fsx/ns1`:
- `configs/single_gpu_full.yaml`
- `configs/resnet_strikes_back.yaml`

You can now start training! 🚀
