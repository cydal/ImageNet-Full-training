#!/bin/bash
# Comprehensive FSx mount diagnostics

echo "=========================================="
echo "FSx Mount Diagnostics"
echo "=========================================="
echo ""

echo "=== System Info ==="
uname -r
lsb_release -ds
echo ""

echo "=== Lustre Client Status ==="
echo "mount.lustre: $(command -v mount.lustre || echo 'NOT FOUND')"
echo "Lustre in /proc/filesystems:"
grep lustre /proc/filesystems 2>/dev/null || echo "  NOT FOUND"
echo "Lustre modules loaded:"
lsmod | grep lustre || echo "  NONE"
echo ""

echo "=== Mount Status ==="
echo "Is /fsx a mountpoint?"
mountpoint /fsx && echo "  YES" || echo "  NO"
echo ""
echo "Mount entries:"
mount | grep -E '(fsx|lustre)' || echo "  No FSx/Lustre mounts found"
echo ""
echo "df for /fsx:"
df -h /fsx 2>/dev/null || echo "  Not mounted"
echo ""

echo "=== Directory Contents ==="
echo "/fsx:"
ls -la /fsx 2>/dev/null || echo "  Cannot list"
echo ""
echo "/fsx/ns1:"
ls -la /fsx/ns1 2>/dev/null || echo "  Cannot list"
echo ""

echo "=== FSx Filesystem Info ==="
echo "DNS: fs-0aba0b7beacfbc7bc.fsx.us-east-1.amazonaws.com"
echo "Mount name: uechfbuv"
echo "DRA path: /ns1 -> s3://sij-imagenet-train/imagenet/"
echo ""

echo "=== S3 Contents ==="
aws s3 ls s3://sij-imagenet-train/imagenet/ 2>/dev/null || echo "  Cannot list S3"
echo ""

echo "=========================================="
echo "Suggested Actions:"
echo "=========================================="
if ! grep -q lustre /proc/filesystems 2>/dev/null; then
    echo "1. Lustre modules not loaded - run mount script or install manually"
elif ! mountpoint -q /fsx; then
    echo "1. Lustre ready but not mounted - run: sudo ./ImageNet-Full-training/scripts/mount_fsx.sh"
else
    echo "1. FSx appears mounted - check /fsx/ns1/ for your data"
fi
