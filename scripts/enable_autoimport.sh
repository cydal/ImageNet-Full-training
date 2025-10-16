#!/bin/bash
# Enable auto-import on the DRA so future S3 changes appear automatically

set -e

DRA_ID="dra-073ac64971b191c89"

echo "=========================================="
echo "Enabling Auto-Import for DRA"
echo "=========================================="
echo ""

echo "Current configuration:"
aws fsx describe-data-repository-associations --association-ids $DRA_ID \
    --query 'Associations[0].S3.AutoImportPolicy' \
    --output json

echo ""
echo "Enabling auto-import for NEW, CHANGED, and DELETED events..."

aws fsx update-data-repository-association \
    --association-id $DRA_ID \
    --s3 'AutoImportPolicy={Events=[NEW,CHANGED,DELETED]}'

echo ""
echo "✓ Auto-import enabled!"
echo ""
echo "New configuration:"
aws fsx describe-data-repository-associations --association-ids $DRA_ID \
    --query 'Associations[0].S3.AutoImportPolicy' \
    --output json

echo ""
echo "=========================================="
echo "Future S3 changes will now appear automatically in FSx"
echo "=========================================="
