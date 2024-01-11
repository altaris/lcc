import warnings
from itertools import chain
from pathlib import Path
from typing import Any, Iterable, Literal

import pytorch_lightning as pl
import torch
import torchmetrics
from loguru import logger as logging
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle
from torchvision.models import get_model
from tqdm import tqdm

from nlnas.clustering import (
    class_otm_matching,
    louvain_communities,
    louvain_loss,
)

from .separability import gdv, label_variation, mean_ggd
from .utils import best_device


class ClusterCorrectionTorchvisionClassifier(TorchvisionClassifier):
    cc_optimizer: torch.optim.Optimizer

    # pylint: disable=unused-argument
    def __init__(
        self,
        model_name: str,
        n_classes: int,
        input_shape: Iterable[int] | None = None,
        model_config: dict[str, Any] | None = None,
        add_final_fc: bool = False,
        cc_lr: float = 1e-4,
        **kwargs,
    ) -> None:
        super().__init__(
            model_name,
            n_classes,
            input_shape,
            model_config,
            add_final_fc,
            **kwargs,
        )
        self.save_hyperparameters()

    def configure_optimizers(self):
        self.cc_optimizer = torch.optim.SGD(
            list(
                chain(
                    *[
                        self.get_submodule(sm).parameters()
                        for sm in self.sep_submodules
                    ]
                )
            ),
            lr=self.hparams["cc_lr"],
        )
        return super().configure_optimizers()

    def on_train_epoch_end(self) -> None:
        """Override"""
        tdl = self.trainer.train_dataloader
        if tdl is None:
            logging.warning(
                "Module's training does not have a train_dataloader"
            )
            return
        # TODO: use params of tdl
        dl = DataLoader(dataset=tdl.dataset, batch_size=2048)
        progress = tqdm(dl, desc="Cluster correction", leave=False)
        for x, y_true in progress:
            out: dict[str, Tensor] = {}
            self.forward_intermediate(
                x, self.sep_submodules, out, keep_gradients=True
            )
            losses: list[Tensor] = []
            for z in out.values():
                _, y_louvain = louvain_communities(z)
                # For testing
                # y_louvain = torch.randint_like(y_true, high=15).cpu().numpy()
                matching = class_otm_matching(y_true.numpy(), y_louvain)
                losses.append(
                    louvain_loss(z, y_true.numpy(), y_louvain, matching)
                )
            loss = torch.stack(losses).mean()
            loss.backward()
            self.cc_optimizer.step()
            self.cc_optimizer.zero_grad()
            progress.set_postfix({"train/lou": float(loss.round(decimals=2))})


class TruncatedClassifier(Classifier):
    """
    Given a `Classifier` and the name of one of its submodule, wrapping it in a
    `TruncatedClassifier` produces a "truncated model" that does the following:
    * evaluate an input `x` on the base classifier,
    * take the latent representation outputed by the specified submodule,
    * flatten it and pass it through a final (trainable) head.
    """

    model: TorchvisionClassifier
    fc: nn.Module
    handle: RemovableHandle
    _out: Tensor

    def __init__(
        self,
        model: TorchvisionClassifier | str | Path,
        truncate_after: str,
        freeze_base_model: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            model (TorchvisionClassifier | str | Path):
            truncate_after (str): e.g. `model.0.classifier.4`
            freeze_base_model (bool, optional):
        """
        _model = (
            model
            if isinstance(model, nn.Module)
            else TorchvisionClassifier.load_from_checkpoint(str(model))
        )  # Need to make sure the model is loaded to get hparams first
        n_classes = _model.hparams["n_classes"]
        input_shape = _model.hparams["input_shape"]
        kwargs["n_classes"] = n_classes

        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.model = _model
        self.fc = _make_lazy_linear(n_classes).to(self.model.device)
        self.model.requires_grad_(not freeze_base_model)

        def _hook(_model: nn.Module, _args: Any, output: Tensor) -> None:
            self._out = output

        submodule = self.model.get_submodule(truncate_after)
        self.handle = submodule.register_forward_hook(_hook)

        self.example_input_array = torch.zeros([1] + list(input_shape))
        self.model.eval()
        self.forward(self.example_input_array)

    # pylint: disable=arguments-differ
    def forward(self, x: Tensor, *_, **__) -> Tensor:
        """Override"""
        self.model(x.to(self.device))  #  type: ignore
        return self.fc(self._out.flatten(1))
