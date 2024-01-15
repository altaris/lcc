"""
A torchvision image classifier wrapped inside a
[`LightningModule`](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html)
"""

import warnings
from typing import Any, Iterable, Literal

import pytorch_lightning as pl
import torch
import torchmetrics
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle
from torchvision.models import get_model

from .clustering import louvain_loss
from .separability import gdv, label_variation, mean_ggd
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
    """Classifier model with some extra features"""

    n_classes: int
    cor_submodules: list[str]
    cor_type: Literal["gdv", "lv", "ggd", "louvain"] | None
    cor_weight: float

    def __init__(
        self,
        n_classes: int,
        cor_submodules: list[str] | None = None,
        cor_type: Literal["gdv", "lv", "ggd", "louvain"] | None = None,
        cor_weight: float = 1e-1,
        sep_submodules: list[str] | None = None,  # Don't use
        sep_score: Literal["gdv", "lv", "ggd", "louvain"]
        | None = None,  # Don't use
        sep_weight: float | None = None,  # Don't use
        **kwargs: Any,
    ) -> None:
        """
        Args:
            n_classes (int):
            cor_submodules (list[str] | None, optional): Submodules to consider
                for the latent correction loss
            cor_type (Literal["gdv", "lv", "gdd"], optional): Type of
                correction, either
                - `gdv` for the Generalized Discrimination Value (see
                  `nlnas.separability.gdv`),
                - `lv` for Label Variation (see `nlnas.separability.lv`),
                - `ggd` for Geodesic Grassmanian Distance
                  (see `nlnas.separability.ggd`),
                - `louvain` for the Louvain/Leiden clustering loss (see
                  `nlnas.clustering.louvain_loss`).
            cor_weight (float, optional): Weight of the correction loss.
                Ignored if `cor_submodules` is left to `None` or is `[]`
            sep_submodules: For backward compatibility with old model
                checkpoints, do not use.
            sep_score: For backward compatibility with old model checkpoints,
                do not use.
            sep_weight: For backward compatibility with old model checkpoints,
                do not use.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters(
            ignore=["sep_submodules", "sep_score", "sep_weight"]
        )
        self.n_classes = n_classes
        self.cor_submodules = (
            (cor_submodules or [])
            if sep_submodules is None
            else sep_submodules
        )
        self.cor_type = cor_type if sep_score is None else sep_score
        self.cor_weight = cor_weight if sep_weight is None else sep_weight

    def _evaluate(self, batch, stage: str | None = None) -> Tensor:
        """Self-explanatory"""
        x, y = batch
        y = y.to(self.device)

        out: dict[str, Tensor] = {}
        logits = self.forward_intermediate(
            x, self.cor_submodules, out, keep_gradients=True
        )
        loss_ce = nn.functional.cross_entropy(logits, y.long())

        compute_cor_loss = (
            stage == "train" and self.cor_type and self.cor_submodules
        )
        if compute_cor_loss:
            if self.cor_type == "gdv":
                loss_sep = torch.stack(
                    [gdv(v, y) for v in out.values()]
                ).mean()
            elif self.cor_type == "lv":
                loss_sep = torch.stack(
                    [
                        label_variation(v, y, k=10, n_classes=self.n_classes)
                        for v in out.values()
                    ]
                ).mean()
            elif self.cor_type == "ggd":
                loss_sep = -torch.stack(
                    [mean_ggd(v.flatten(1), y) for v in out.values()]
                ).mean()
            elif self.cor_type == "louvain":
                loss_sep = torch.stack(
                    [louvain_loss(v, y) for v in out.values()]
                ).mean()
            else:
                raise RuntimeError(
                    f"Unknown correction type '{self.cor_type}'."
                )
        else:
            loss_sep = torch.tensor(0.0)

        if stage:
            self.log(f"{stage}/loss", loss_ce, prog_bar=True, sync_dist=True)
        if compute_cor_loss:
            self.log(
                f"{stage}/{self.cor_type}",
                loss_sep,
                prog_bar=True,
                sync_dist=True,
            )
        if stage and best_device() != "mps":
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
            self.log(f"{stage}/acc", acc, prog_bar=True, sync_dist=True)
        return loss_ce + self.cor_weight * loss_sep

    def configure_optimizers(self):
        """Override"""
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
        """Override"""
        return self._evaluate(batch, "test")

    # pylint: disable=arguments-differ
    def training_step(self, batch, *_, **__) -> Any:
        """Override"""
        return self._evaluate(batch, "train")

    def validation_step(self, batch, *_, **__):
        """Override"""
        return self._evaluate(batch, "val")


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
        """Override"""
        return self.model(x.to(self.device))  # type: ignore
