"""
ImageNet LightningDataModule supporting tiny/medium/full datasets.
Supports logical subsetting for flexible testing without creating separate directories.
"""
import os
import random
from pathlib import Path
from typing import Optional

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms


class RemappedSubset(Dataset):
    """
    Subset of a dataset with remapped labels to [0, N-1].
    
    This is necessary when using a subset of ImageNet classes, as the original
    labels may be sparse (e.g., classes 5, 17, 234) but the model expects
    dense labels (0, 1, 2).
    """
    def __init__(self, dataset, indices, label_mapping=None):
        """
        Args:
            dataset: Original dataset
            indices: Indices to include in subset
            label_mapping: Dict mapping original labels to new labels [0, N-1]
                          If None, labels are used as-is
        """
        self.dataset = dataset
        self.indices = indices
        self.label_mapping = label_mapping
    
    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        if self.label_mapping is not None:
            label = self.label_mapping[label]
        return image, label
    
    def __len__(self):
        return len(self.indices)


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
        prefetch_factor: Optional[int] = None,
        random_crop: bool = True,
        random_horizontal_flip: bool = True,
        auto_augment: Optional[str] = None,
        random_erasing: float = 0.0,  # Random Erasing probability (RSB A2 uses 0.25)
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
        self.prefetch_factor = prefetch_factor
        self.img_size = img_size
        self.mean = mean
        self.std = std
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.random_crop = random_crop
        self.random_horizontal_flip = random_horizontal_flip
        self.auto_augment = auto_augment
        self.random_erasing = random_erasing
        
        # Subsetting parameters
        self.max_classes = max_classes
        self.max_samples_per_class = max_samples_per_class
        self.subset_seed = subset_seed
        
        self.train_dataset = None
        self.val_dataset = None
        self._num_classes = None  # Will be set during setup
    
    def _create_subset(self, dataset, max_classes=None, max_samples_per_class=None):
        """
        Create a logical subset of the dataset with remapped labels.
        
        Args:
            dataset: ImageFolder dataset
            max_classes: Maximum number of classes to include (None = all)
            max_samples_per_class: Maximum samples per class (None = all)
        
        Returns:
            RemappedSubset or original dataset if no subsetting requested
        
        Note:
            CRITICAL: When both parameters are None (full training), this function
            immediately returns the original dataset with ZERO overhead. Full training
            is completely unaffected by the subsetting code.
        """
        # IMPORTANT: Early return for full training - no modifications, no overhead
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
        
        # Create label remapping: original_label -> new_label [0, N-1]
        label_mapping = {}
        for new_label, original_label in enumerate(sorted(class_to_indices.keys())):
            label_mapping[original_label] = new_label
        
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
        print(f"Label mapping: {len(label_mapping)} original classes -> [0, {len(label_mapping)-1}]")
        
        # Store number of classes for model initialization
        if not hasattr(self, '_num_classes') or self._num_classes is None:
            self._num_classes = len(class_to_indices)
        
        return RemappedSubset(dataset, subset_indices, label_mapping)
    
    def setup(self, stage: Optional[str] = None):
        """Setup train and validation datasets with optional subsetting."""
        if stage == "fit" or stage is None:
            train_transform = self._get_train_transform()
            full_train = datasets.ImageFolder(
                root=self.data_root / "train",
                transform=train_transform
            )
            # Store original number of classes before subsetting
            if self.max_classes is None:
                self._num_classes = len(full_train.classes)
            
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
    
    def num_classes(self) -> int:
        """Get the number of classes in the dataset.
        
        Must be called after setup() has been run.
        """
        if self._num_classes is None:
            raise RuntimeError(
                "num_classes() called before setup(). "
                "Either call setup() first or provide max_classes during initialization."
            )
        return self._num_classes
    
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
        
        # Random Erasing (RSB A2 uses this)
        if self.random_erasing > 0:
            transform_list.append(
                transforms.RandomErasing(p=self.random_erasing, scale=(0.02, 0.33), ratio=(0.3, 3.3))
            )
        
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
        loader_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": True,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers if self.num_workers > 0 else False,
            "drop_last": True
        }
        if self.prefetch_factor is not None and self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(self.train_dataset, **loader_kwargs)
    
    def val_dataloader(self):
        """Return validation dataloader."""
        loader_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers if self.num_workers > 0 else False,
            "drop_last": False
        }
        if self.prefetch_factor is not None and self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(self.val_dataset, **loader_kwargs)
