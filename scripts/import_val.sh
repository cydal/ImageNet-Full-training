#!/bin/bash
# Import metadata for val/ folder from S3 to FSx

set -e

echo "=========================================="
echo "Importing val/ metadata from S3 to FSx"
echo "=========================================="
echo ""

FS_ID="fs-0aba0b7beacfbc7bc"
DRA_ID="dra-073ac64971b191c89"

echo "Filesystem: $FS_ID"
echo "DRA: $DRA_ID"
echo "Path: /ns1/val/"
echo ""

# Check current DRA status
echo "Current DRA configuration:"
aws fsx describe-data-repository-associations --association-ids $DRA_ID \
    --query 'Associations[0].{Path:FileSystemPath,S3:DataRepositoryPath,AutoImport:S3.AutoImportPolicy}' \
    --output table

echo ""
echo "Starting import task for val/ directory..."
echo ""

# Start import task for the entire DRA
# This will re-import metadata from S3, including val/
TASK_ID=$(aws fsx create-data-repository-task \
    --type IMPORT_METADATA_FROM_REPOSITORY \
    --file-system-id $FS_ID \
    --report Enabled=false \
    --query 'DataRepositoryTask.TaskId' \
    --output text)

echo "Import task created: $TASK_ID"
echo ""
echo "Monitoring task status (this may take a few minutes)..."
echo ""

# Monitor task
while true; do
    STATUS=$(aws fsx describe-data-repository-tasks \
        --task-ids $TASK_ID \
        --query 'DataRepositoryTasks[0].Lifecycle' \
        --output text)
    
    if [ "$STATUS" == "SUCCEEDED" ]; then
        echo "✓ Import completed successfully!"
        break
    elif [ "$STATUS" == "FAILED" ] || [ "$STATUS" == "CANCELED" ]; then
        echo "✗ Import failed with status: $STATUS"
        aws fsx describe-data-repository-tasks --task-ids $TASK_ID
        exit 1
    else
        echo "  Status: $STATUS (waiting...)"
        sleep 5
    fi
done

echo ""
echo "Checking /fsx/ns1/ contents..."
ls -la /fsx/ns1/

echo ""
echo "=========================================="
echo "Import complete!"
echo "=========================================="
