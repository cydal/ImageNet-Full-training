# Configuration Files

## Training Configs

- `base.yaml` - Base configuration with common settings
- `local.yaml` - Local development configuration
- `tiny.yaml` - Tiny ImageNet subset for testing
- `tiny_gpu.yaml` - Tiny subset optimized for GPU testing
- `full.yaml` - Full ImageNet training configuration

## FSx Configuration

### `fsx_config.example.sh`
Template configuration file with placeholder values. **Safe to commit to git.**

### `fsx_config.sh` (gitignored)
Your actual FSx configuration with real filesystem IDs. **DO NOT commit this file.**

#### First Time Setup

```bash
# 1. Copy the example
cp configs/fsx_config.example.sh configs/fsx_config.sh

# 2. Edit with your actual values
nano configs/fsx_config.sh

# 3. Update these values:
#    - FSX_DNS_NAME: Your FSx DNS endpoint
#    - FSX_MOUNT_NAME: Your mount name (or leave commented for auto-detect)
#    - FSX_ID: Your filesystem ID
#    - TRAIN_DATA_PATH: Path to training data
#    - VAL_DATA_PATH: Path to validation data
```

#### Usage

```bash
# Source the config before running scripts
source configs/fsx_config.sh

# Then run mount or other FSx scripts
./scripts/mount_fsx.sh
```

#### Security Note

`fsx_config.sh` is automatically ignored by git (see `.gitignore`). This prevents accidentally committing your actual AWS filesystem IDs and paths to version control.

If you need to share configuration across team members:
1. Each person creates their own `fsx_config.sh` from the example
2. Update the `.example` file with comments/documentation, not real values
3. Share filesystem IDs through secure channels (not git)
