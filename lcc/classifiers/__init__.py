"""Classifier models and related stuff"""

from .base import (
    BaseClassifier,
    LatentClusteringData,
    validate_lcc_kwargs,
)
from .huggingface import HuggingFaceClassifier
from .timm import TimmClassifier
from .torchvision import TorchvisionClassifier
from .wrapped import WrappedClassifier

__all__ = [
    "BaseClassifier",
    "get_classifier_cls",
    "HuggingFaceClassifier",
    "LatentClusteringData",
    "TimmClassifier",
    "TorchvisionClassifier",
    "validate_lcc_kwargs",
    "WrappedClassifier",
]


def get_classifier_cls(model_name: str) -> type[BaseClassifier]:
    """
    Returns the classifier class to use for a given model name.

    Args:
        model_name (str):
    """
    if model_name.startswith("timm/"):
        return TimmClassifier
    if "/" in model_name:
        return HuggingFaceClassifier
    return TorchvisionClassifier
