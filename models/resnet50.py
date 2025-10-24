"""
ResNet50 model factory with Lightning wrapper.
Wraps torchvision ResNet50 with training logic.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import lightning.pytorch as pl
from torchmetrics.functional import accuracy
from copy import deepcopy
from torchvision.models import resnet50, ResNet50_Weights
from typing import Optional, Dict, Any
from pytorch_lamb import Lamb

from utils.metrics import accuracy

torch.set_float32_matmul_precision('medium')


class SoftTargetCrossEntropy(nn.Module):
    """
    Soft target cross entropy loss for mixup/cutmix.
    Expects soft targets (probabilities) instead of hard labels.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x, target):
        """
        Args:
            x: logits (B, C)
            target: soft targets (B, C) - probabilities that sum to 1
        """
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class BCEWithLogitsLossForMixup(nn.Module):
    """
    BCE loss for mixup/cutmix (RSB A2 uses this).
    Converts soft targets to multi-label format and uses BCE.
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, x, target):
        """
        Args:
            x: logits (B, C)
            target: soft targets (B, C) - probabilities that sum to 1
        """
        return self.bce(x, target)


class ResNet50Module(pl.LightningModule):
    """
    Lightning wrapper for ResNet50 with configurable training.
    """
    
    def __init__(
        self,
        num_classes: int = 1000,
        pretrained: bool = False,
        optimizer: str = "sgd",
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        lr_scheduler: str = "cosine",
        warmup_epochs: int = 5,
        max_epochs: int = 90,
        cosine_t_max: int = None,  # If None, uses max_epochs - warmup_epochs
        eta_min: float = 1e-6,  # Minimum LR for cosine annealing (never reach 0)
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        label_smoothing: float = 0.0,
        compile_model: bool = False,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model
        if pretrained:
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.model = resnet50(weights=None)
        
        # Modify final layer if num_classes != 1000
        if num_classes != 1000:
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        # Compile model for faster training (PyTorch 2.0+)
        if compile_model:
            self.model = torch.compile(self.model, mode='max-autotune')
        
        # Use channels_last memory format for better Tensor Core utilization
        self.model = self.model.to(memory_format=torch.channels_last)
        
        # Loss function
        # Use soft-target cross entropy for mixup/cutmix (RSB A2)
        self.use_mixup = (mixup_alpha > 0 or cutmix_alpha > 0)
        if self.use_mixup:
            self.criterion = SoftTargetCrossEntropy()  # Soft-target CE for mixup/cutmix
            self.val_criterion = nn.CrossEntropyLoss()  # Standard CE for validation (hard labels)
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            self.val_criterion = self.criterion
        
        # Metrics
        self.train_acc1_sum = 0.0
        self.train_acc5_sum = 0.0
        self.train_samples = 0
        
        # EMA (Exponential Moving Average) for better evaluation
        self.ema_model = None
        self.ema_decay = 0.9999
    
    def forward(self, x):
        return self.model(x)
    
    def on_load_checkpoint(self, checkpoint):
        """
        Handle loading modified checkpoints that have optimizer/scheduler states removed.
        This allows continuation training with new hyperparameters.
        """
        # Lightning will handle missing optimizer_states and lr_schedulers gracefully
        # by reinitializing them from the new hyperparameters
        pass
    
    def training_step(self, batch, batch_idx):
        images, targets = batch
        
        # Convert to channels_last for better performance
        images = images.to(memory_format=torch.channels_last)
        
        # Apply mixup/cutmix if enabled
        if self.hparams.mixup_alpha > 0 or self.hparams.cutmix_alpha > 0:
            images, targets = self._apply_mixup_cutmix(images, targets)
            outputs = self(images)
            loss = self._mixup_criterion(outputs, targets)
        else:
            outputs = self(images)
            loss = self.criterion(outputs, targets)
        
        # Compute accuracy (only for non-mixed targets)
        if self.hparams.mixup_alpha == 0 and self.hparams.cutmix_alpha == 0:
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            self.train_acc1_sum += acc1.item() * images.size(0)
            self.train_acc5_sum += acc5.item() * images.size(0)
            self.train_samples += images.size(0)
        
        # Log loss and learning rate
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log current learning rate
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("train/lr", current_lr, on_step=True, on_epoch=False, prog_bar=False)
        
        # Update EMA model
        self._update_ema()
        
        return loss
    
    def on_train_epoch_end(self):
        """Log epoch-level training metrics."""
        if self.train_samples > 0:
            avg_acc1 = self.train_acc1_sum / self.train_samples
            avg_acc5 = self.train_acc5_sum / self.train_samples
            self.log("train/acc1", avg_acc1, prog_bar=True)
            self.log("train/acc5", avg_acc5)
            
            # Reset
            self.train_acc1_sum = 0.0
            self.train_acc5_sum = 0.0
            self.train_samples = 0
    
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        
        # Convert to channels_last for better performance
        images = images.to(memory_format=torch.channels_last)
        
        # Use the main model for validation (EMA disabled for debugging)
        # TODO: Re-enable EMA once we verify training works
        outputs = self(images)
        
        # Compute loss - use validation criterion (handles hard labels)
        loss = self.val_criterion(outputs, targets)
        
        # Compute accuracy (always use hard labels)
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        
        # Log
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/acc1", acc1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/acc5", acc5, on_step=False, on_epoch=True, sync_dist=True)
        
        return loss
    
    def _apply_mixup_cutmix(self, images, targets):
        """Apply mixup or cutmix augmentation."""
        # Simplified: randomly choose between mixup and cutmix
        # In practice, you might want to use timm's Mixup class
        import random
        
        if random.random() < 0.5 and self.hparams.mixup_alpha > 0:
            # Mixup
            lam = torch.distributions.Beta(self.hparams.mixup_alpha, self.hparams.mixup_alpha).sample()
            index = torch.randperm(images.size(0), device=images.device)
            mixed_images = lam * images + (1 - lam) * images[index]
            targets_a, targets_b = targets, targets[index]
            return mixed_images, (targets_a, targets_b, lam)
        elif self.hparams.cutmix_alpha > 0:
            # CutMix
            lam = torch.distributions.Beta(self.hparams.cutmix_alpha, self.hparams.cutmix_alpha).sample()
            index = torch.randperm(images.size(0), device=images.device)
            
            # Generate random box
            W, H = images.size(2), images.size(3)
            cut_rat = torch.sqrt(1.0 - lam)
            cut_w = (W * cut_rat).int()
            cut_h = (H * cut_rat).int()
            
            cx = torch.randint(W, (1,)).item()
            cy = torch.randint(H, (1,)).item()
            
            bbx1 = torch.clamp(cx - cut_w // 2, 0, W)
            bby1 = torch.clamp(cy - cut_h // 2, 0, H)
            bbx2 = torch.clamp(cx + cut_w // 2, 0, W)
            bby2 = torch.clamp(cy + cut_h // 2, 0, H)
            
            mixed_images = images.clone()
            mixed_images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]
            
            # Adjust lambda
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
            targets_a, targets_b = targets, targets[index]
            return mixed_images, (targets_a, targets_b, lam)
        
        return images, targets
    
    def _mixup_criterion(self, outputs, targets):
        """Compute loss for mixed targets (soft targets)."""
        targets_a, targets_b, lam = targets
        
        # Convert lam to float if it's a tensor
        if isinstance(lam, torch.Tensor):
            lam = lam.item()
        
        # Create soft targets using F.one_hot and manual mixing
        num_classes = outputs.size(1)
        targets_a_onehot = F.one_hot(targets_a, num_classes).float()
        targets_b_onehot = F.one_hot(targets_b, num_classes).float()
        
        # Mix the one-hot targets
        soft_targets = lam * targets_a_onehot + (1.0 - lam) * targets_b_onehot
        
        return self.criterion(outputs, soft_targets)
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Separate parameters: exclude BN and bias from weight decay
        params_with_wd = []
        params_without_wd = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # No weight decay for BN parameters and biases
            if 'bn' in name or 'bias' in name or 'norm' in name:
                params_without_wd.append(param)
            else:
                params_with_wd.append(param)
        
        param_groups = [
            {'params': params_with_wd, 'weight_decay': self.hparams.weight_decay},
            {'params': params_without_wd, 'weight_decay': 0.0}
        ]
        
        # Optimizer
        if self.hparams.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(
                param_groups,
                lr=self.hparams.lr,
                momentum=self.hparams.momentum
            )
        elif self.hparams.optimizer.lower() == "lamb":
            # LAMB optimizer (RSB A2 uses this)
            optimizer = Lamb(
                param_groups,
                lr=self.hparams.lr,
                betas=(0.9, 0.999),
                eps=1e-6
            )
        elif self.hparams.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(
                param_groups,
                lr=self.hparams.lr
            )
        elif self.hparams.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.hparams.lr
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer}")
        
        # Learning rate scheduler
        if self.hparams.lr_scheduler.lower() == "cosine":
            # Cosine annealing after warmup
            # Use cosine_t_max if specified, otherwise use max_epochs - warmup_epochs
            t_max = self.hparams.cosine_t_max if self.hparams.cosine_t_max is not None else (self.hparams.max_epochs - self.hparams.warmup_epochs)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=t_max,
                eta_min=self.hparams.eta_min
            )
        elif self.hparams.lr_scheduler.lower() == "step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[30, 60, 80],
                gamma=0.1
            )
        else:
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        
        # Warmup
        if self.hparams.warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=self.hparams.warmup_epochs
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[self.hparams.warmup_epochs]
            )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val/loss"
            }
        }
    
    def _update_ema(self):
        """Update EMA model weights."""
        if self.ema_model is None:
            # Initialize EMA model on first call
            # Don't copy compiled model - copy the underlying model
            if hasattr(self.model, '_orig_mod'):
                # torch.compile wraps the model
                self.ema_model = deepcopy(self.model._orig_mod)
            else:
                self.ema_model = deepcopy(self.model)
            self.ema_model.eval()
            self.ema_model = self.ema_model.to(self.device)
            for param in self.ema_model.parameters():
                param.requires_grad = False
        else:
            # Update EMA weights: ema = decay * ema + (1 - decay) * model
            with torch.no_grad():
                # Get the actual model parameters (unwrap if compiled)
                if hasattr(self.model, '_orig_mod'):
                    model_params = self.model._orig_mod.parameters()
                else:
                    model_params = self.model.parameters()
                
                for ema_param, model_param in zip(self.ema_model.parameters(), model_params):
                    ema_param.data.mul_(self.ema_decay).add_(model_param.data, alpha=1 - self.ema_decay)


def create_model(config: Dict[str, Any]) -> ResNet50Module:
    """Factory function to create ResNet50 model from config."""
    return ResNet50Module(**config)
