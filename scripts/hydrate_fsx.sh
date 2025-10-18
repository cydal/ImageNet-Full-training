#!/bin/bash
# Hydrate FSx cache by reading all training data
# This forces FSx to fetch data from S3 and cache it locally

set -e

echo "================================================================================"
echo "FSx Cache Hydration Script"
echo "================================================================================"
echo ""
echo "This script will read all ImageNet data from S3 into FSx cache."
echo "This is a ONE-TIME operation that will take 2-6 hours."
echo "After completion, training will be 1000x faster!"
echo ""

# Configuration
DATA_DIRS=("/fsx/ns1/train" "/fsx/ns1/val")
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/fsx_hydration.log"
PARALLEL_JOBS=32  # Number of parallel read operations

# Create log directory
mkdir -p "$LOG_DIR"

# Function to hydrate a directory
hydrate_directory() {
    local dir=$1
    local name=$(basename "$dir")
    
    echo "================================================================================"
    echo "Hydrating: $dir"
    echo "================================================================================"
    
    # Count total files
    echo "Counting files..."
    total_files=$(find "$dir" -type f -name "*.JPEG" | wc -l)
    echo "Total files to hydrate: $total_files"
    echo ""
    
    # Start hydration
    echo "Starting hydration with $PARALLEL_JOBS parallel jobs..."
    echo "This will take a while. Progress is logged to: $LOG_FILE"
    echo ""
    
    start_time=$(date +%s)
    
    # Use find + xargs to read files in parallel
    # Reading to /dev/null forces FSx to cache the data
    find "$dir" -type f -name "*.JPEG" -print0 | \
        xargs -0 -P "$PARALLEL_JOBS" -I {} sh -c 'cat {} > /dev/null' 2>&1 | \
        tee -a "$LOG_FILE"
    
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    elapsed_min=$((elapsed / 60))
    
    echo ""
    echo "✓ Completed: $dir"
    echo "  Files hydrated: $total_files"
    echo "  Time taken: ${elapsed}s (${elapsed_min} minutes)"
    echo "  Average: $(echo "scale=2; $total_files / $elapsed" | bc) files/sec"
    echo ""
}

# Main execution
echo "$(date): Starting FSx cache hydration" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

overall_start=$(date +%s)

# Hydrate each directory
for dir in "${DATA_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        hydrate_directory "$dir"
    else
        echo "⚠️  Warning: Directory not found: $dir" | tee -a "$LOG_FILE"
    fi
done

overall_end=$(date +%s)
overall_elapsed=$((overall_end - overall_start))
overall_elapsed_min=$((overall_elapsed / 60))
overall_elapsed_hr=$((overall_elapsed / 3600))

echo "================================================================================"
echo "FSx Cache Hydration Complete!"
echo "================================================================================"
echo ""
echo "Total time: ${overall_elapsed}s (${overall_elapsed_min} min / ${overall_elapsed_hr} hr)"
echo ""
echo "$(date): Hydration complete!" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "Next steps:"
echo "  1. Verify cache with: python benchmark_data_loading.py"
echo "  2. Expected FSx read speed: 200-300 MB/s (was 0.1 MB/s)"
echo "  3. Start training with fast data loading!"
echo ""
echo "FSx cache will persist across training runs."
echo "You only need to run this once (unless FSx is recreated)."
echo ""
