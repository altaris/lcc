"""See `WrappedClassifier` documentation."""

from typing import Any

from torch import Tensor, nn

from nlnas.logging import r0_debug

from .base import BaseClassifier, Batch


class WrappedClassifier(BaseClassifier):
    """
    An image classifier model
    ([`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html))
    wrapped inside a `nlnas.classifiers.BaseClassifier`.
    """

    model: nn.Module
    logit_key: str | None

    def __init__(
        self,
        model: nn.Module,
        n_classes: int,
        head_name: str | None = None,
        logit_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        See also:
            `nlnas.classifiers.BaseClassifier.__init__`.

        Args:
            model (nn.Module):
            n_classes (int): Number of classes in the dataset on which the model
                will be trained.
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
            logit_key (str | None, optional): If the wrapped model outputs a
                dict-like object instead of a tensor, this key is used to access
                the actual logits.
        """
        self.save_hyperparameters(ignore=["model"])
        super().__init__(n_classes, **kwargs)
        self.model, self.logit_key = model, logit_key
        if head_name:
            replace_head(self.model, head_name, n_classes)

    def forward(self, inputs: Tensor | Batch, *_: Any, **__: Any) -> Tensor:
        x: Tensor = (
            inputs if isinstance(inputs, Tensor) else inputs[self.image_key]
        )
        output = self.model(x.to(self.device))
        return output if self.logit_key is None else output[self.logit_key]  # type: ignore

    @property
    def lcc_submodules(self) -> list[str]:
        """
        Returns the list of submodules considered for LCC, whith correct prefix
        if needed.
        """
        return (
            []
            if not self.hparams["lcc_submodules"]
            else [
                (sm if sm.startswith("model.") else "model." + sm)
                for sm in self.hparams["lcc_submodules"]
            ]
        )

    def on_train_start(self) -> None:
        self.model.train()
        super().on_train_start()


def replace_head(
    module: nn.Module, head_name: str, n_classes: int
) -> nn.Module:
    """
    Replaces the last linear layer of a model with a new one with a specified
    number of output neurons (which is not necessarily different from the old
    head's). *However*, if the model's head already has the correct number of
    output neurons, then it is not replaced.

    Args:
        module (nn.Module):
        head_name (str): e.g. `model.classifier.1`. The
            name of a submodule can be retried by inspecting the output of
            `nn.Module.named_modules` or `nlnas.utils.pretty_print_submodules`.
        n_classes (int): The desired number of output neurons.

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
    if head.out_features == n_classes:
        r0_debug(
            "Module head '{}' already have {} output neurons. "
            "Not replacing it",
            head_name,
            n_classes,
        )
        return module
    new_head = nn.Linear(
        in_features=head.in_features,
        out_features=n_classes,
        bias=head.bias is not None,
    )
    parent = module.get_submodule(".".join(head_name.split(".")[:-1]))
    if isinstance(parent, nn.Sequential):
        parent[-1] = new_head
    else:
        setattr(parent, head_name.split(".")[-1], new_head)
    return module
