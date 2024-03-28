"""An image classifier wrapped inside a `BaseClassifier`"""

# pylint: disable=too-many-ancestors

from typing import Any

from torch import Tensor, nn

from .base import BaseClassifier, Batch


class WrappedClassifier(BaseClassifier):
    """See module documentation"""

    model: nn.Module

    def __init__(
        self,
        model: nn.Module,
        n_classes: int,
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
        super().__init__(n_classes, **kwargs)
        self.save_hyperparameters(ignore=["model"])
        self.model = model

    # pylint: disable=arguments-differ
    # pylint: disable=missing-function-docstring
    def forward(self, inputs: Tensor | Batch, *_, **__) -> Tensor:
        x: Tensor = (
            inputs if isinstance(inputs, Tensor) else inputs[self.image_key]
        )
        output = self.model(x.to(self.device))  # type: ignore
        return output if self.logit_key is None else output[self.logit_key]
