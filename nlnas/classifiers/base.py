"""Base image classifier class that support clustering correction loss"""

from typing import Any

import pytorch_lightning as pl
import torch
import torchmetrics
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from ..clustering import louvain_loss
from ..utils import best_device


class BaseClassifier(pl.LightningModule):
    """See module documentation"""

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
