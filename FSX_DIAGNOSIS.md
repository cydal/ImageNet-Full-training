# FSx Performance Diagnosis

## Problem Identified

**FSx read speed is catastrophically slow: 0.1 MB/s (should be ~300 MB/s)**

### Benchmark Results

```
Raw FSx Read Speed:
  Read 100 files (13.5 MB) in 94.15s
  Throughput: 0.1 MB/s  ← 3000x slower than expected!
  Files/sec: 1.1

JPEG Decoding (CPU):
  Decoded 100 JPEGs in 0.52s
  Images/sec: 192.8  ← This is FAST, not the bottleneck
```

### FSx Configuration

```json
{
    "StorageCapacity": 1200 GB,
    "PerUnitStorageThroughput": 250 MB/s per TiB,
    "Total Throughput": ~300 MB/s,
    "DeploymentType": "PERSISTENT_2",
    "DataCompressionType": "NONE"
}
```

**Expected:** 300 MB/s  
**Actual:** 0.1 MB/s  
**Slowdown:** **3000x**

## Root Cause

### FSx Lazy Loading from S3

FSx for Lustre with S3 integration uses **lazy loading**:

1. **First access:** File metadata exists, but data is in S3
2. **On read:** FSx fetches data from S3 (slow!)
3. **Cached:** Subsequent reads are fast from FSx cache

**Current state:** Files are NOT cached in FSx, every read goes to S3

### Why This Happens

- Auto-import only imports **metadata**, not data
- Data is fetched from S3 on first access
- S3 read latency: ~100-200ms per file
- ImageNet has 1.2M small files = catastrophic for training

## Solutions

### Solution 1: Pre-load Data into FSx Cache (RECOMMENDED)

**Hydrate the FSx cache by reading all files once:**

```bash
# Create hydration script
cat > /home/ubuntu/ImageNet-Full-training/scripts/hydrate_fsx.sh << 'EOF'
#!/bin/bash
# Hydrate FSx cache by reading all training data

echo "Starting FSx cache hydration..."
echo "This will take several hours but only needs to be done once"
echo ""

DATA_DIR="/fsx/ns1/train"
LOG_FILE="logs/fsx_hydration.log"

mkdir -p logs

echo "$(date): Starting hydration of $DATA_DIR" | tee -a $LOG_FILE

# Use find + xargs to read files in parallel
# This forces FSx to cache data from S3
find "$DATA_DIR" -type f -name "*.JPEG" | \
    xargs -P 32 -I {} sh -c 'cat {} > /dev/null' 2>&1 | \
    tee -a $LOG_FILE

echo "$(date): Hydration complete!" | tee -a $LOG_FILE
echo "FSx cache is now populated. Training will be fast." | tee -a $LOG_FILE
EOF

chmod +x /home/ubuntu/ImageNet-Full-training/scripts/hydrate_fsx.sh
```

**Run hydration:**
```bash
# This will take 2-6 hours depending on FSx throughput
nohup ./scripts/hydrate_fsx.sh &

# Monitor progress
tail -f logs/fsx_hydration.log
```

**After hydration:**
- FSx cache will be populated
- Subsequent reads will be ~300 MB/s (fast!)
- Training will achieve expected GPU utilization

### Solution 2: Use FSx Data Repository Task (Alternative)

```bash
# Export all data from S3 to FSx
aws fsx create-data-repository-task \
    --file-system-id fs-02386cb09beeabb62 \
    --type EXPORT_TO_REPOSITORY \
    --paths /ns1/train /ns1/val \
    --report Enabled=false

# Then import to force caching
aws fsx create-data-repository-task \
    --file-system-id fs-02386cb09beeabb62 \
    --type IMPORT_METADATA_FROM_REPOSITORY \
    --report Enabled=false
```

**Note:** This may not actually cache the data, just metadata.

### Solution 3: Increase FSx Throughput (Expensive)

```bash
# Increase to 500 MB/s per TiB (2x current)
aws fsx update-file-system \
    --file-system-id fs-02386cb09beeabb62 \
    --lustre-configuration PerUnitStorageThroughput=500
```

**Cost:** ~2x more expensive  
**Benefit:** Faster S3→FSx transfer, but still slow on first access

### Solution 4: Copy Data to Local NVMe (Fast but Limited)

If instance has local NVMe storage:

```bash
# Copy to local NVMe (if available)
sudo mkfs.ext4 /dev/nvme1n1
sudo mkdir -p /data
sudo mount /dev/nvme1n1 /data

# Copy data (will take time but only once)
rsync -av --progress /fsx/ns1/ /data/imagenet/

# Update config to use /data/imagenet
```

**Pros:** Extremely fast (NVMe speed)  
**Cons:** Limited space, data lost on instance termination

## Recommended Action Plan

### Step 1: Hydrate FSx Cache (Do This Now)

```bash
cd /home/ubuntu/ImageNet-Full-training

# Create hydration script
./scripts/hydrate_fsx.sh &

# Monitor progress
watch -n 10 'tail -20 logs/fsx_hydration.log'
```

**Expected time:** 2-6 hours  
**One-time cost:** Yes, data stays cached in FSx

### Step 2: Verify Cache is Populated

After hydration, re-run benchmark:

```bash
python benchmark_data_loading.py
```

**Expected results after hydration:**
```
Raw FSx Read Speed:
  Throughput: 200-300 MB/s  ← Should be 2000-3000x faster!
  Files/sec: 1000-2000
```

### Step 3: Resume Training

After FSx cache is populated:

```bash
python train.py \
    --config configs/single_gpu_full.yaml \
    --wandb_project imagenet-resnet50 \
    --wandb_name single-gpu-cached-fsx
```

**Expected performance:**
- GPU utilization: 80-95%
- Training speed: 1.5-3.0 it/s
- Time per epoch: 30-60 minutes

## Why This Wasn't Obvious

1. **Auto-import is misleading** - Only imports metadata, not data
2. **FSx appears mounted** - Files are visible but data is in S3
3. **No clear error** - Just slow, not broken
4. **First-time issue** - After cache is populated, problem disappears

## Multi-Node Implications

**Critical for multi-node training:**

- **Without cache:** All nodes will hit S3 simultaneously (even slower!)
- **With cache:** All nodes read from FSx cache (fast and scalable)

**Before multi-node training:**
1. ✅ Hydrate FSx cache completely
2. ✅ Verify with benchmark
3. ✅ Test single-node performance
4. ✅ Then scale to multi-node

## Verification Checklist

After hydration:

- [ ] Re-run `benchmark_data_loading.py`
- [ ] FSx read speed > 200 MB/s
- [ ] DataLoader throughput > 1.0 batches/sec
- [ ] GPU utilization > 80%
- [ ] Training speed > 1.5 it/s

## Summary

**Problem:** FSx lazy-loading from S3 (0.1 MB/s)  
**Solution:** Hydrate FSx cache (one-time, 2-6 hours)  
**Result:** Fast training (300 MB/s, 80-95% GPU utilization)

**This is a one-time setup cost.** Once FSx cache is populated, training will be fast and the cache persists across training runs.
