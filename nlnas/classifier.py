"""A torchvision image classifier wrapped inside a `LightningModule`"""

from itertools import chain
from typing import Any, Iterable, Tuple

import pytorch_lightning as pl
import torch
import torchmetrics
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle
from torchvision.models import get_model
from tqdm import tqdm

from .separability import gdv, label_variation
from .utils import best_device


class Classifier(pl.LightningModule):
    """Classifier model with some extra features"""

    n_classes: int

    def __init__(self, n_classes: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.n_classes = n_classes

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
            keep_gradients (bool): If `True`, the tensors in `output_dict` keep
                their gradients (if they had some on the first place). If
                `False`, they are detached and moved to the CPU.
        """

        def create_hook(key: str):
            def hook(_model: nn.Module, _args: Any, output: Tensor) -> None:
                output_dict[key] = (
                    output if keep_gradients else output.detach().cpu()
                )

            return hook

        handles: list[RemovableHandle] = []
        for name in submodules:
            submodule = self.get_submodule(name)
            handles.append(submodule.register_forward_hook(create_hook(name)))
        y = self(x)
        for h in handles:
            h.remove()
        return y

    def test_step(self, batch, *_, **__):
        return self._evaluate(batch, "test")

    # pylint: disable=arguments-differ
    def training_step(self, batch, *_, **__) -> Any:
        return self._evaluate(batch, "train")

    def validation_step(self, batch, *_, **__):
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
        modules = [get_model(model_name, **(model_config or {}))]
        if add_final_fc:
            modules.append(nn.LazyLinear(n_classes))
        self.model = nn.Sequential(*modules)
        if input_shape is not None:
            self.example_input_array = torch.zeros([1] + list(input_shape))
            self.model.eval()
            self.forward(self.example_input_array)

    # pylint: disable=arguments-differ
    def forward(self, x: Tensor, *_, **__) -> Tensor:
        return self.model(x.to(self.device))  # type: ignore


class VHTorchvisionClassifier(TorchvisionClassifier):
    """Torchvision classifier with a horizontal training hook"""

    vh_submodules: list[str]

    def __init__(
        self,
        model_name: str,
        n_classes: int,
        vh_submodules: list[str],
        input_shape: Iterable[int] | None = None,
        model_config: dict[str, Any] | None = None,
        add_final_fc: bool = False,
        horizontal_lr: float = 1e-3,
        **kwargs,
    ) -> None:
        """
        Args:
            model_name (str): See `nlnas.classifier.TorchvisionClassifier`
            n_classes (int): See `nlnas.classifier.TorchvisionClassifier`
            vh_submodules (list[str]): Names of the submodules that are to be
                trained horizontally (as well as vertically)
            input_shape (Iterable[int] | None, optional): See
                `nlnas.classifier.TorchvisionClassifier`
            model_config (dict[str, Any] | None, optional): See
                `nlnas.classifier.TorchvisionClassifier`
            add_final_fc (bool, optional): See
                `nlnas.classifier.TorchvisionClassifier`
        """
        super().__init__(
            model_name,
            n_classes,
            input_shape,
            model_config,
            add_final_fc,
            **kwargs,
        )
        self.save_hyperparameters()
        self.vh_submodules = vh_submodules
        self.horizontal_opt = torch.optim.Adam(
            chain(
                *[
                    self.get_submodule(s).parameters()
                    for s in self.vh_submodules
                ]
            ),
            lr=horizontal_lr,
        )
        # self.automatic_optimization = False

    def on_train_epoch_end(self) -> None:
        dl = self.trainer.train_dataloader
        if dl is None:
            raise RuntimeError(
                "Model's trainer does not have a training dataloader. "
                "This should really not happen =("
            )
        progress = tqdm(iter(dl), desc="Horizontal fit", leave=False)
        for batch in progress:
            self.horizontal_opt.zero_grad()
            x, y = batch
            output_dict: dict[str, Tensor] = {}
            self.forward_intermediate(
                x, self.vh_submodules, output_dict, keep_gradients=True
            )
            loss = torch.mean(
                torch.stack([gdv(v, y) for v in output_dict.values()])
                # torch.stack(
                #     [
                #         label_variation(
                #             v,
                #             y.to(v),
                #             k=10,
                #             n_classes=self.n_classes,
                #         )
                #         for v in output_dict.values()
                #     ]
                # )
            )
            loss.backward()
            self.horizontal_opt.step()
            # progress.set_postfix({"train/lv": float(loss)})
            progress.set_postfix({"train/gdv": float(loss)})
            # self.log("train/lv", loss, sync_dist=True)
            self.log("train/gdv", loss, sync_dist=True)
