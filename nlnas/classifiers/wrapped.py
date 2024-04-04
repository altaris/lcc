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
        head_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            model (nn.Module):
            n_classes (int): Number of classes
            head_name (str | None, optional): Name of the head submodule in
                `model`, which is the fully connected (aka
                [`nn.Linear`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html))
                layer that outputs the logits. If not `None`, the head is
                replaced by a new fully connected layer with `n_classes` output
                neurons classes. Specify this to fine-tune a pretrained model
                on a new dataset with the same or a different number of
                classes. The name of a submodule can be retried by inspecting
                the output of `nn.Module.named_modules` or
                `nlnas.utils.pretty_print_submodules`.
            kwargs: Forwarded to `BaseClassifier`
        """
        self.save_hyperparameters(ignore=["model"])
        super().__init__(n_classes, **kwargs)
        self.model = model
        if head_name:
            replace_head(self.model, head_name, n_classes)

    # pylint: disable=arguments-differ
    # pylint: disable=missing-function-docstring
    def forward(self, inputs: Tensor | Batch, *_, **__) -> Tensor:
        x: Tensor = (
            inputs if isinstance(inputs, Tensor) else inputs[self.image_key]
        )
        output = self.model(x.to(self.device))  # type: ignore
        return output if self.logit_key is None else output[self.logit_key]


def replace_head(
    module: nn.Module, head_name: str, out_features: int
) -> nn.Module:
    """
    Replaces the last linear layer of a model with a new one with a specified
    number of output neurons (which is not necessarily different from the old
    head's).

    Args:
        module (nn.Module):
        head_name (str): e.g. `model.classifier.1`. The
            name of a submodule can be retried by inspecting the output of
            `nn.Module.named_modules` or `nlnas.utils.pretty_print_submodules`.
        out_features (int):

    Raises:
        RuntimeError: If the head module is not a
        [`nn.Linear`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
        module

    Returns:
        The modified module, which is the one passed in argument since the
        modification is performed in-place.
    """
    head = module.get_submodule(head_name)
    if not isinstance(head, nn.Linear):
        raise RuntimeError(
            f"Model head '{head_name}' must have type nn.Linear"
        )
    new_head = nn.Linear(
        in_features=head.in_features,
        out_features=out_features,
        bias=not (head.bias is None),
    )
    parent = module.get_submodule(".".join(head_name.split(".")[:-1]))
    if isinstance(parent, nn.Sequential):
        parent[-1] = new_head
    else:
        setattr(parent, head_name.split(".")[-1], new_head)
    return module
