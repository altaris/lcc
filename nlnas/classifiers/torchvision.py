"""See `TorchvisionClassifier` documentation."""

from typing import Any, Callable

import torch
import torchvision.transforms.v2 as transforms
from torchvision.models import get_model

from .wrapped import WrappedClassifier


class TorchvisionClassifier(WrappedClassifier):
    """
    A torchvision classifier wrapped as a `WrappedClassifier`. See
    https://pytorch.org/vision/stable/models.html#classification .
    """

    def __init__(
        self,
        model_name: str,
        n_classes: int,
        head_name: str | None = None,
        weights: Any = "DEFAULT",
        **kwargs: Any,
    ) -> None:
        model = get_model(model_name, weights=weights)
        super().__init__(model, n_classes, head_name, **kwargs)
        self.save_hyperparameters()

    @staticmethod
    def get_image_processor(
        model_name: str, **__: Any
    ) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """
        Torchvision models do not require an image processor.
        """

        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RGB(),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
            ]
        )

        def _processor(batch: dict[str, Any]) -> dict[str, Any]:
            return {
                k: (
                    transform(v)
                    # TODO: pass image_key from DS ↓
                    if k in ["img", "image", "jpg", "png"]
                    else v
                )
                for k, v in batch.items()
            }

        return _processor
