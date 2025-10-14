"""
ResNet50 model factory with Lightning wrapper.
Wraps torchvision ResNet50 with training logic.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import resnet50, ResNet50_Weights
from typing import Optional, Dict, Any

from utils.metrics import accuracy


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
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        label_smoothing: float = 0.0,
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
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # Metrics
        self.train_acc1_sum = 0.0
        self.train_acc5_sum = 0.0
        self.train_samples = 0
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, targets = batch
        
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
        
        # Log
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
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
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        
        # Compute accuracy
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
        """Compute loss for mixed targets."""
        targets_a, targets_b, lam = targets
        loss_a = self.criterion(outputs, targets_a)
        loss_b = self.criterion(outputs, targets_b)
        return lam * loss_a + (1 - lam) * loss_b
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Optimizer
        if self.hparams.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.lr,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer}")
        
        # Learning rate scheduler
        if self.hparams.lr_scheduler.lower() == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.max_epochs - self.hparams.warmup_epochs,
                eta_min=0
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
                "frequency": 1
            }
        }


def create_model(config: Dict[str, Any]) -> ResNet50Module:
    """Factory function to create ResNet50 model from config."""
    return ResNet50Module(**config)
