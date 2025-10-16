#!/bin/bash
# Import metadata from S3 to FSx for Lustre
# Use this if directories/files are missing after mounting FSx

set -e

FSX_ID="${FSX_ID:-}"
PATHS="${PATHS:-}"

echo "=========================================="
echo "FSx Metadata Import Tool"
echo "=========================================="
echo ""

# Check if FSX_ID is provided
if [[ -z "$FSX_ID" ]]; then
    # Try to detect from mount
    if mountpoint -q /fsx 2>/dev/null; then
        MOUNT_INFO=$(mount | grep "on /fsx type lustre")
        if [[ -n "$MOUNT_INFO" ]]; then
            echo "Detected FSx mount at /fsx"
            # Try to extract from DNS if available
            FSX_DNS=$(echo "$MOUNT_INFO" | grep -oP '\S+@tcp' | cut -d'@' -f1)
            echo "Mount source: $FSX_DNS"
        fi
    fi
    
    # Still need user input
    if [[ -z "$FSX_ID" ]]; then
        echo "Error: FSX_ID environment variable is required"
        echo ""
        echo "Usage:"
        echo "  export FSX_ID=fs-xxxxx"
        echo "  $0"
        echo ""
        echo "Or provide it inline:"
        echo "  FSX_ID=fs-xxxxx $0"
        echo ""
        echo "Optional: Specify paths to import (relative to DRA FileSystemPath)"
        echo "  FSX_ID=fs-xxxxx PATHS='[\"subdir/\"]' $0"
        exit 1
    fi
fi

echo "Filesystem ID: $FSX_ID"
echo ""

# Check for existing import tasks
echo "Checking for running import tasks..."
RUNNING_TASKS=$(aws fsx describe-data-repository-tasks \
    --filters Name=file-system-id,Values="$FSX_ID" \
    --query 'DataRepositoryTasks[?Lifecycle==`EXECUTING` || Lifecycle==`PENDING`].[TaskId,Lifecycle]' \
    --output text 2>/dev/null || echo "")

if [[ -n "$RUNNING_TASKS" ]]; then
    echo "Warning: Found running/pending import tasks:"
    echo "$RUNNING_TASKS"
    echo ""
    read -p "Continue and create another task? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

# Show DRA info
echo "Data Repository Associations:"
aws fsx describe-data-repository-associations \
    --filters Name=file-system-id,Values="$FSX_ID" \
    --query 'Associations[*].{Path:FileSystemPath,S3:DataRepositoryPath,AutoImport:S3.AutoImportPolicy.Events}' \
    --output table 2>/dev/null || echo "No DRAs found"

echo ""
echo "Starting metadata import task..."

# Build command
CMD="aws fsx create-data-repository-task \
    --type IMPORT_METADATA_FROM_REPOSITORY \
    --file-system-id $FSX_ID \
    --report Enabled=false"

if [[ -n "$PATHS" ]]; then
    echo "Importing specific paths: $PATHS"
    CMD="$CMD --paths '$PATHS'"
else
    echo "Importing all DRA paths"
fi

# Execute
TASK_ID=$(eval "$CMD --query 'DataRepositoryTask.TaskId' --output text")

if [[ -z "$TASK_ID" ]]; then
    echo "Error: Failed to create import task"
    exit 1
fi

echo "✓ Import task created: $TASK_ID"
echo ""
echo "Monitoring progress (Ctrl+C to stop monitoring, task will continue)..."
echo ""

# Monitor
while true; do
    TASK_INFO=$(aws fsx describe-data-repository-tasks \
        --task-ids "$TASK_ID" \
        --query 'DataRepositoryTasks[0].{Status:Lifecycle,Progress:Status.SucceededCount,Failed:Status.FailedCount}' \
        --output json 2>/dev/null)
    
    STATUS=$(echo "$TASK_INFO" | jq -r '.Status')
    
    if [[ "$STATUS" == "SUCCEEDED" ]]; then
        echo "✓ Import completed successfully!"
        echo "$TASK_INFO" | jq .
        break
    elif [[ "$STATUS" == "FAILED" ]] || [[ "$STATUS" == "CANCELED" ]]; then
        echo "✗ Import failed with status: $STATUS"
        aws fsx describe-data-repository-tasks --task-ids "$TASK_ID"
        exit 1
    else
        echo "$(date +%H:%M:%S) - Status: $STATUS"
        sleep 5
    fi
done

echo ""
echo "=========================================="
echo "Import complete!"
echo "=========================================="
echo ""
echo "If FSx is mounted, verify your data is now visible:"
echo "  ls -la /fsx/"
