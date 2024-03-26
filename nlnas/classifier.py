"""
A torchvision image classifier wrapped inside a
[`LightningModule`](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html)
"""

import warnings
from typing import Any, Iterable

import pytorch_lightning as pl
import torch
import torchmetrics
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle
from torchvision.models import get_model

from .clustering import louvain_loss
from .utils import best_device


def _make_lazy_linear(*args, **kwargs) -> nn.LazyLinear:
    """
    Constructs a
    [`LazyLinear`](https://pytorch.org/docs/stable/generated/torch.nn.LazyLinear.html)
    layer but hides the following warning:

        [...]torch/nn/modules/lazy.py:180: UserWarning: Lazy modules are a new
        feature under heavy development so changes to the API or functionality
        can happen at any moment.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return nn.LazyLinear(*args, **kwargs)


class Classifier(pl.LightningModule):
    """Abstract classifier model with some extra features"""

    n_classes: int
    cor_submodules: list[str]
    cor_weight: float
    cor_kwargs: dict[str, Any]

    def __init__(
        self,
        n_classes: int,
        cor_submodules: list[str] | None = None,
        cor_weight: float = 1e-1,
        cor_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            n_classes (int):
            cor_submodules (list[str] | None, optional): Submodules to consider
                for the latent correction loss
            cor_weight (float, optional): Weight of the correction loss.
                Ignored if `cor_submodules` is left to `None` or is `[]`
            cor_kwargs (dict, optional): Passed to the correction loss function
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.n_classes = n_classes
        self.cor_submodules = cor_submodules or []
        self.cor_weight, self.cor_kwargs = cor_weight, cor_kwargs or {}

    def _evaluate(self, batch, stage: str | None = None) -> Tensor:
        """Self-explanatory"""
        x, y = batch
        y = y.to(self.device)

        out: dict[str, Tensor] = {}
        logits = self.forward_intermediate(
            x, self.cor_submodules, out, keep_gradients=True
        )
        loss_ce = nn.functional.cross_entropy(logits, y.long())

        compute_correction_loss = stage == "train" and self.cor_submodules
        if compute_correction_loss:
            loss_lou = torch.stack(
                [louvain_loss(v, y, **self.cor_kwargs) for v in out.values()]
            ).mean()
        else:
            loss_lou = torch.tensor(0.0)

        log: dict[str, Tensor] = {}
        if stage:
            log[f"{stage}/loss"] = loss_ce
            if best_device() != "mps":
                # NotImplementedError: The operator 'aten::_unique2' is not
                # currently implemented for the MPS device. If you want this op to
                # be added in priority during the prototype phase of this feature,
                # please comment on
                # https://github.com/pytorch/pytorch/issues/77764. As a temporary
                # fix, you can set the environment variable
                # `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for
                # this op. WARNING: this will be slower than running natively on
                # MPS.
                acc = torchmetrics.functional.accuracy(
                    torch.argmax(logits, dim=1),
                    y,
                    "multiclass",
                    num_classes=self.n_classes,
                    top_k=1,
                )
                log[f"{stage}/acc"] = acc
        if compute_correction_loss:
            log[f"{stage}/louvain"] = loss_lou
        self.log_dict(log, prog_bar=True, sync_dist=True)
        return loss_ce + self.cor_weight * loss_lou

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def forward_intermediate(
        self,
        x: Tensor,
        submodules: list[str],
        output_dict: dict,
        keep_gradients: bool = False,
    ) -> Tensor:
        """
        Runs the model and collects the output of specified submodules. The
        intermediate outputs are stored in `output_dict` under the
        corresponding submodule name. In particular, this method has side
        effects.

        Args:
            x (Tensor):
            submodules (list[str]):
            output_dict (dict):
            keep_gradients (bool, optional): If `True`, the tensors in
                `output_dict` keep their gradients (if they had some on the
                first place). If `False`, they are detached and moved to the
                CPU.
        """

        def create_hook(key: str):
            def hook(_model: nn.Module, _args: Any, output: Tensor) -> None:
                x = output if keep_gradients else output.cpu().detach()
                output_dict[key] = x

            return hook

        handles: list[RemovableHandle] = []
        for name in submodules:
            submodule = self.get_submodule(name)
            handles.append(submodule.register_forward_hook(create_hook(name)))
        y = self.forward(x)
        for h in handles:
            h.remove()
        return y

    # pylint: disable=arguments-differ
    def test_step(self, batch, *_, **__):
        return self._evaluate(batch, "test")

    # pylint: disable=arguments-differ
    def training_step(self, batch, *_, **__) -> Any:
        return self._evaluate(batch, "train")

    def validation_step(self, batch, *_, **__):
        return self._evaluate(batch, "val")


# pylint: disable=too-many-ancestors
class TorchvisionClassifier(Classifier):
    """
    A torchvision image classifier with some extra features

    See also:
        https://pytorch.org/vision/stable/models.html#classification
    """

    model: nn.Module

    def __init__(
        self,
        model_name: str,
        n_classes: int,
        input_shape: Iterable[int] | None = None,
        model_config: dict[str, Any] | None = None,
        add_final_fc: bool = False,
        **kwargs,
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
            add_final_fc (bool): If true, adds a final dense layer which
                outputs `n_classes` logits
        """
        super().__init__(n_classes=n_classes, **kwargs)
        self.save_hyperparameters()
        model_config = model_config or {}
        if "num_classes" not in model_config:
            model_config["num_classes"] = n_classes
        modules = [get_model(model_name, **model_config)]
        if add_final_fc:
            modules.append(_make_lazy_linear(n_classes))
        self.model = nn.Sequential(*modules)
        if input_shape is not None:
            self.example_input_array = torch.zeros([1] + list(input_shape))
            self.model.eval()
            self.forward(self.example_input_array)

    # pylint: disable=arguments-differ
    def forward(self, x: Tensor, *_, **__) -> Tensor:
        return self.model(x.to(self.device))  # type: ignore
