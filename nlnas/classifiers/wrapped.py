"""An image classifier wrapped inside a `BaseClassifier`"""

# pylint: disable=too-many-ancestors

from typing import Any

from torch import Tensor, nn

from .base import BaseClassifier


class WrappedClassifier(BaseClassifier):
    """See module documentation"""

    model: nn.Module

    def __init__(
        self,
        model: nn.Module,
        n_classes: int,
        cor_submodules: list[str] | None = None,
        cor_weight: float = 0.1,
        cor_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            model (nn.Module):
            n_classes (int):
            cor_submodules (list[str] | None, optional): See `BaseClassifier`
            cor_weight (float, optional): See `BaseClassifier`
            cor_kwargs (dict[str, Any] | None, optional): See `BaseClassifier`
        """
        super().__init__(
            n_classes, cor_submodules, cor_weight, cor_kwargs, **kwargs
        )
        self.save_hyperparameters(ignore=["model"])
        self.model = model

    # pylint: disable=arguments-differ
    def forward(self, x: Tensor, *_, **__) -> Tensor:
        return self.model(x.to(self.device))  # type: ignore
