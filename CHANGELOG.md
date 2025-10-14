# Changelog

## [Unreleased] - 2025-10-14

### Changed - Import Updates
- **Updated all imports from `pytorch_lightning` to `lightning.pytorch`**
  - This follows the new Lightning 2.0+ package structure
  - Affected files:
    - `data/datamodule.py`
    - `models/resnet50.py`
    - `utils/callbacks.py`
    - `train.py`
    - `eval.py`
    - `tests/quick_test.py`
  
- **Updated `requirements.txt`**
  - Removed: `pytorch-lightning>=2.0.0`
  - Kept: `lightning>=2.0.0` (correct package name)

### Added - Documentation Clarity
- **Created `SETUP.md`**
  - Clear environment setup instructions
  - Installation verification steps
  - Common issues and solutions
  - Emphasizes that setup is REQUIRED before running tests

- **Updated `tests/test_plan_datamodule.md`**
  - Added prominent warning that tests are NOT YET EXECUTED
  - Clarified distinction between "test code implemented" vs "test executed"
  - Added Phase 0: Environment Setup as prerequisite
  - All test sections now clearly show:
    - ✅ Test code implemented
    - ⏳ NOT YET EXECUTED (requires environment setup)

- **Updated `README.md`**
  - Added prominent "Setup Required" section at top
  - Links to SETUP.md for detailed instructions
  - Emphasizes installation is required first

### Fixed - Test Status Clarity
- **Before**: Tests were marked as "completed" (✅) which was misleading
- **After**: Tests clearly show:
  - ✅ = Test code exists
  - ⏳ = Not yet executed (waiting for environment setup)
  
This addresses the confusion about how tests could be "complete" without environment setup.

## Migration Guide

### For Users
If you're starting fresh:
1. Run `make install` to set up environment
2. Verify with `python -c "import lightning.pytorch as pl; print(pl.__version__)"`
3. Run tests: `make quick-test`

### For Developers
If you have existing code using old imports:
```python
# OLD (deprecated)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger

# NEW (correct)
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
```

## Summary

### What Changed
1. ✅ All imports updated to `lightning.pytorch`
2. ✅ Requirements.txt cleaned up
3. ✅ Test status clarified (code exists, not yet run)
4. ✅ Setup documentation added
5. ✅ README updated with setup requirements

### What's Next
1. ⏳ Install dependencies: `make install`
2. ⏳ Run tests: `make quick-test`
3. ⏳ Verify data module: `make test-data`
4. ⏳ Check data integrity: `make test-integrity`
5. ⏳ Benchmark performance: `make benchmark-data`
