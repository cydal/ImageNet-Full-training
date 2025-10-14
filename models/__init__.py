"""Models for ImageNet training."""
from .resnet50 import ResNet50Module, create_model

__all__ = ["ResNet50Module", "create_model"]
