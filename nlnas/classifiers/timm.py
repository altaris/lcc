"""
Pretrained classifier model loaded from the [HuggingFace model
hub](https://huggingface.co/models?pipeline_tag=image-classification) uploaded
by the `timm` team.
"""

from typing import Any, Callable

import timm
from torch import Tensor

from .wrapped import Batch, WrappedClassifier


class TimmClassifier(WrappedClassifier):
    """See module documentation."""

    image_transform: Callable

    def __init__(
        self,
        model_name: str,
        n_classes: int,
        head_name: str | None = None,
        pretrained: bool = True,
        **kwargs: Any,
    ) -> None:
        model = timm.create_model(model_name, pretrained=pretrained)
        super().__init__(model, n_classes, head_name, **kwargs)
        self.save_hyperparameters()

    def image_processor(self, is_training: bool = True) -> Callable:
        """Returns an image processor for the model"""
        config = timm.data.resolve_model_data_config(self.model)
        return timm.data.create_transform(**config, is_training=is_training)

    # pylint: disable=arguments-differ
    # pylint: disable=missing-function-docstring
    def forward(self, inputs: Tensor | Batch, *_, **__) -> Tensor:
        x: Tensor = (
            inputs if isinstance(inputs, Tensor) else inputs[self.image_key]
        )
        return self.model(x.to(self.device))  # type: ignore
