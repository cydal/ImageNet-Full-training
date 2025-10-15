"""
ImageNet LightningDataModule supporting tiny/medium/full datasets.
Supports logical subsetting for flexible testing without creating separate directories.
"""
import os
import random
from pathlib import Path
from typing import Optional

import lightning.pytorch as pl
from torch.utils.data import DataLoader, Subset
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
    
    Logical Subsetting:
        Use max_classes and max_samples_per_class to create a subset
        without physically creating a separate directory.
        
        Example:
            # Use only 10 classes with 100 samples each
            dm = ImageNetDataModule(
                data_root="/data/imagenet",
                max_classes=10,
                max_samples_per_class=100
            )
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
        # Logical subsetting parameters
        max_classes: Optional[int] = None,
        max_samples_per_class: Optional[int] = None,
        subset_seed: int = 42,
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
        
        # Subsetting parameters
        self.max_classes = max_classes
        self.max_samples_per_class = max_samples_per_class
        self.subset_seed = subset_seed
        
        self.train_dataset = None
        self.val_dataset = None
    
    def _create_subset(self, dataset, max_classes=None, max_samples_per_class=None):
        """
        Create a logical subset of the dataset.
        
        Args:
            dataset: ImageFolder dataset
            max_classes: Maximum number of classes to include (None = all)
            max_samples_per_class: Maximum samples per class (None = all)
        
        Returns:
            Subset of the dataset or original dataset if no subsetting requested
        """
        if max_classes is None and max_samples_per_class is None:
            return dataset
        
        # Build class-to-indices mapping
        class_to_indices = {}
        for idx, (_, label) in enumerate(dataset.samples):
            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)
        
        # Limit number of classes if requested
        if max_classes is not None:
            random.seed(self.subset_seed)
            all_classes = sorted(class_to_indices.keys())
            selected_classes = random.sample(
                all_classes, 
                min(max_classes, len(all_classes))
            )
            class_to_indices = {
                k: v for k, v in class_to_indices.items() 
                if k in selected_classes
            }
        
        # Limit samples per class if requested
        subset_indices = []
        for label, indices in sorted(class_to_indices.items()):
            if max_samples_per_class is not None:
                random.seed(self.subset_seed + label)
                selected_indices = random.sample(
                    indices, 
                    min(max_samples_per_class, len(indices))
                )
            else:
                selected_indices = indices
            subset_indices.extend(selected_indices)
        
        print(f"Created subset: {len(subset_indices)} samples from {len(class_to_indices)} classes")
        return Subset(dataset, subset_indices)
    
    def setup(self, stage: Optional[str] = None):
        """Setup train and validation datasets with optional subsetting."""
        if stage == "fit" or stage is None:
            train_transform = self._get_train_transform()
            full_train = datasets.ImageFolder(
                root=self.data_root / "train",
                transform=train_transform
            )
            # Apply subsetting if requested
            self.train_dataset = self._create_subset(
                full_train,
                max_classes=self.max_classes,
                max_samples_per_class=self.max_samples_per_class
            )
        
        if stage == "fit" or stage == "validate" or stage is None:
            val_transform = self._get_val_transform()
            full_val = datasets.ImageFolder(
                root=self.data_root / "val",
                transform=val_transform
            )
            # Apply subsetting if requested (use fewer val samples)
            val_samples = 10 if self.max_samples_per_class else None
            self.val_dataset = self._create_subset(
                full_val,
                max_classes=self.max_classes,
                max_samples_per_class=val_samples
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
