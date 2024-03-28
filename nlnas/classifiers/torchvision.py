"""
A torchvision image classifier wrapped inside a `BaseClassifier`

See also:
    https://pytorch.org/vision/stable/models.html#classification
"""

# pylint: disable=too-many-ancestors

from typing import Any, Iterable

import torch
from torchvision.models import get_model

from .wrapped import WrappedClassifier


class TorchvisionClassifier(WrappedClassifier):
    """See module documentation"""

    def __init__(
        self,
        model_name: str,
        n_classes: int,
        input_shape: Iterable[int] | None = None,
        model_config: dict[str, Any] | None = None,
        cor_submodules: list[str] | None = None,
        cor_weight: float = 0.1,
        cor_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            model_name (str): Torchvision model name in lower case. See also
                https://pytorch.org/vision/stable/generated/torchvision.models.list_models.html
            n_classes (int):
            input_shape (Iterable[int], optional): If give, a example run is
                performed after construction. This can be useful to see the
                model's computation graph on tensorboard.
            model_config (dict[str, Any], optional):
            cor_submodules (list[str] | None, optional): See `BaseClassifier`
            cor_weight (float, optional): See `BaseClassifier`
            cor_kwargs (dict[str, Any] | None, optional): See `BaseClassifier`
        """
        model_config = model_config or {}
        if "num_classes" not in model_config:
            model_config["num_classes"] = n_classes
        model = get_model(model_name, **model_config)
        super().__init__(
            model, n_classes, cor_submodules, cor_weight, cor_kwargs, **kwargs
        )
        self.save_hyperparameters()
        if input_shape is not None:
            self.example_input_array = torch.zeros([1] + list(input_shape))
            self.model.eval()
            self.forward(self.example_input_array)
