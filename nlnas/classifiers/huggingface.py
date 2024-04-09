"""
Pretrained classifier model loaded from the [HuggingFace model
hub](https://huggingface.co/models?pipeline_tag=image-classification).
"""

from typing import Any

from transformers import AutoModelForImageClassification

from .wrapped import WrappedClassifier


class HuggingFaceClassifier(WrappedClassifier):
    """See module documentation."""

    def __init__(
        self,
        model_name: str,
        n_classes: int,
        head_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        model = AutoModelForImageClassification.from_pretrained(model_name)
        super().__init__(model, n_classes, head_name, **kwargs)
        self.save_hyperparameters()
