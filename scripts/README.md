# Scripts Directory

## FSx for Lustre Scripts

### `mount_fsx.sh`
Mounts AWS FSx for Lustre filesystem with automatic Lustre client setup and mount name detection.

**Usage:**
```bash
export FSX_DNS_NAME=fs-xxxxx.fsx.us-east-1.amazonaws.com
./mount_fsx.sh
```

**Features:**
- Auto-installs Lustre client if needed
- Detects FSx mount name from AWS API
- Checks kernel compatibility
- Shows DRA information

See [FSX_SETUP.md](../docs/FSX_SETUP.md) for detailed documentation.

### `import_fsx_metadata.sh`
Imports metadata from S3 to FSx when directories are missing after mount.

**Usage:**
```bash
export FSX_ID=fs-xxxxx
./import_fsx_metadata.sh
```

**When to use:**
- Directories don't appear after mounting FSx
- Data was uploaded to S3 after FSx/DRA creation
- Need to refresh FSx metadata from S3

## Training Scripts

### `env_setup.sh`
Sets up Python environment and installs dependencies.

### `launch_single.sh`
Launches training on a single GPU instance.

### `launch_multi.sh`
Launches distributed training across multiple nodes.

### `make_tiny_subset.py`
Creates a small subset of ImageNet for testing.

## Configuration

Use the example configuration file:
```bash
cp ../configs/fsx_config.example.sh ../configs/fsx_config.sh
# Edit fsx_config.sh with your values
source ../configs/fsx_config.sh
```
