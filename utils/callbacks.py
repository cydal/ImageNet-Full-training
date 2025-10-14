"""
Custom PyTorch Lightning callbacks.
Includes EMA (Exponential Moving Average) and other utilities.
"""
import copy
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback


class EMACallback(Callback):
    """
    Exponential Moving Average (EMA) callback.
    Maintains a shadow copy of model weights with exponential moving average.
    """
    
    def __init__(self, decay: float = 0.9999, validate_original_weights: bool = False):
        """
        Args:
            decay: EMA decay factor (typically 0.999 or 0.9999)
            validate_original_weights: If True, validate with original weights instead of EMA
        """
        super().__init__()
        self.decay = decay
        self.validate_original_weights = validate_original_weights
        self.ema_state_dict = None
    
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Initialize EMA weights."""
        # Create a copy of the initial model state
        self.ema_state_dict = copy.deepcopy(pl_module.state_dict())
    
    def on_train_batch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        outputs, 
        batch, 
        batch_idx
    ):
        """Update EMA weights after each training batch."""
        if self.ema_state_dict is None:
            return
        
        with torch.no_grad():
            # Get current model state
            model_state_dict = pl_module.state_dict()
            
            # Update EMA weights
            for key in self.ema_state_dict.keys():
                if model_state_dict[key].dtype in [torch.float32, torch.float16, torch.bfloat16]:
                    self.ema_state_dict[key].mul_(self.decay).add_(
                        model_state_dict[key], alpha=1 - self.decay
                    )
                else:
                    # For non-float parameters (e.g., integers), just copy
                    self.ema_state_dict[key] = model_state_dict[key].clone()
    
    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Swap to EMA weights before validation."""
        if self.ema_state_dict is None or self.validate_original_weights:
            return
        
        # Save original weights
        self.original_state_dict = copy.deepcopy(pl_module.state_dict())
        
        # Load EMA weights
        pl_module.load_state_dict(self.ema_state_dict)
    
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Restore original weights after validation."""
        if self.ema_state_dict is None or self.validate_original_weights:
            return
        
        # Restore original weights
        pl_module.load_state_dict(self.original_state_dict)
        del self.original_state_dict
    
    def on_save_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint):
        """Save EMA weights in checkpoint."""
        if self.ema_state_dict is not None:
            checkpoint["ema_state_dict"] = self.ema_state_dict
    
    def on_load_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint):
        """Load EMA weights from checkpoint."""
        if "ema_state_dict" in checkpoint:
            self.ema_state_dict = checkpoint["ema_state_dict"]


class ThroughputMonitor(Callback):
    """
    Monitor training throughput (images/sec).
    """
    
    def __init__(self):
        super().__init__()
        self.batch_start_time = None
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Record batch start time."""
        import time
        self.batch_start_time = time.time()
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Compute and log throughput."""
        import time
        if self.batch_start_time is None:
            return
        
        batch_time = time.time() - self.batch_start_time
        batch_size = batch[0].size(0)
        
        # Account for distributed training
        world_size = trainer.world_size if trainer.world_size else 1
        total_images = batch_size * world_size
        
        throughput = total_images / batch_time
        
        pl_module.log("train/throughput", throughput, on_step=True, on_epoch=False)
