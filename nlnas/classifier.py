"""Abstract image classifier model"""

from typing import Any

import pytorch_lightning as pl
import torch
import torchmetrics
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle
from torchvision.models import get_model

from nlnas.utils import best_device


class Classifier(pl.LightningModule):
    """Convenient module wrapper for image classifiers"""

    model: nn.Module

    n_classes: int

    def __init__(
        self,
        model: nn.Module,
        n_classes: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model, self.n_classes = model, n_classes

    def _evaluate(self, batch, stage: str | None = None) -> Tensor:
        """Self-explanatory"""
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y.long())
        if stage:
            self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True)
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
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    # pylint: disable=arguments-differ
    def forward(self, x: Tensor, *_, **__) -> Tensor:
        return self.model(x.to(self.device))  # type: ignore

    def forward_intermediate(
        self,
        x: Tensor,
        submodules: list[str],
        output_dict: dict,
    ) -> None:
        """
        Runs the model and collects the output of specified submodules. The
        intermediate outputs are stored in `output_dict` under the
        corresponding submodule name. In particular, this method has side
        effects.

        Args:
            x (Tensor):
            submodules (list[str]):
            output_dict (dict):
        """

        def create_hook(key: str):
            def hook(_model: nn.Module, _args: Any, output: Tensor) -> None:
                output_dict[key] = output.detach().cpu()

            return hook

        handles: list[RemovableHandle] = []
        for name in submodules:
            submodule = self.get_submodule(name)
            handles.append(submodule.register_forward_hook(create_hook(name)))
        self(x)
        for h in handles:
            h.remove()

    def test_step(self, batch, *_, **__):
        return self._evaluate(batch, "test")

    # pylint: disable=arguments-differ
    def training_step(self, batch, *_, **__) -> Any:
        x, y = batch
        loss = nn.functional.cross_entropy(self.model(x), y.long())
        self.log("train/loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, *_, **__):
        return self._evaluate(batch, "val")

    @staticmethod
    def torchvision_classifier(
        name: str,
        n_classes: int,
        config: dict[str, Any] | None = None,
        add_final_fc: bool = False,
    ):
        """
        Instanciates and wraps a torchvision image classifier. See also
        https://pytorch.org/vision/stable/models.html#classification and

        Args:
            name (str):
            config (dict[str, Any] | None, optional): Passed to the model class
                at construction
            n_classes (int):
            add_final_fc (bool):
        """
        modules = [get_model(name, **(config or {}))]
        if add_final_fc:
            modules.append(nn.LazyLinear(n_classes))
        return Classifier(nn.Sequential(*modules), n_classes)
