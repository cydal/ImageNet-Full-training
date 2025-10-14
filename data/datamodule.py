"""
ImageNet LightningDataModule supporting tiny/medium/full datasets.
"""
import os
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class ImageNetDataModule(pl.LightningDataModule):
    """
    LightningDataModule for ImageNet dataset.
    
    Supports standard ImageNet directory structure:
        data_root/
            train/
                n01440764/
                    n01440764_10026.JPEG
                    ...
                n01443537/
                    ...
            val/
                n01440764/
                    ILSVRC2012_val_00000293.JPEG
                    ...
    """
    
    def __init__(
        self,
        data_root: str,
        batch_size: int = 256,
        num_workers: int = 8,
        img_size: int = 224,
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225),
        pin_memory: bool = True,
        persistent_workers: bool = True,
        random_crop: bool = True,
        random_horizontal_flip: bool = True,
        auto_augment: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.mean = mean
        self.std = std
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.random_crop = random_crop
        self.random_horizontal_flip = random_horizontal_flip
        self.auto_augment = auto_augment
        
        self.train_dataset = None
        self.val_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """Setup train and validation datasets."""
        if stage == "fit" or stage is None:
            train_transform = self._get_train_transform()
            self.train_dataset = datasets.ImageFolder(
                root=self.data_root / "train",
                transform=train_transform
            )
        
        if stage == "fit" or stage == "validate" or stage is None:
            val_transform = self._get_val_transform()
            self.val_dataset = datasets.ImageFolder(
                root=self.data_root / "val",
                transform=val_transform
            )
    
    def _get_train_transform(self):
        """Build training data augmentation pipeline."""
        transform_list = []
        
        # Random resized crop
        if self.random_crop:
            transform_list.append(
                transforms.RandomResizedCrop(
                    self.img_size,
                    scale=(0.08, 1.0),
                    interpolation=transforms.InterpolationMode.BILINEAR
                )
            )
        else:
            transform_list.extend([
                transforms.Resize(int(self.img_size * 256 / 224)),
                transforms.CenterCrop(self.img_size)
            ])
        
        # Random horizontal flip
        if self.random_horizontal_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        
        # AutoAugment
        if self.auto_augment == "imagenet":
            transform_list.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET))
        elif self.auto_augment == "randaugment":
            transform_list.append(transforms.RandAugment())
        
        # To tensor and normalize
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        return transforms.Compose(transform_list)
    
    def _get_val_transform(self):
        """Build validation data preprocessing pipeline."""
        # Standard ImageNet validation: resize to 256, center crop to 224
        resize_size = int(self.img_size * 256 / 224)
        
        return transforms.Compose([
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
    
    def train_dataloader(self):
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            drop_last=True
        )
    
    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            drop_last=False
        )
