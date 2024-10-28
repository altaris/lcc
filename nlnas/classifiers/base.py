"""Base image classifier class that support clustering correction loss"""

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Sequence, TypeAlias

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed
from timm.loss import BinaryCrossEntropy
from timm.optim import create_optimizer_v2
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle
from torchmetrics.functional.classification import multiclass_accuracy

from nlnas.correction.clustering import lcc_loss

from ..utils import (
    to_array,
)
from .utils import (
    log_optimizers_lr,
    temporary_directory,
    validate_lcc_kwargs,
)

Batch: TypeAlias = dict[str, Tensor]


@dataclass
class LatentClusteringData:
    """
    Convenience struct that holds some latent clustering correction data for a
    given latent space.
    """

    p: np.ndarray
    """
    `(N,)` boolean vector that marks misclustered samples (regardless of
    true class).
    """

    matching: dict[int, set[int]]
    """
    Matching between the true and cluster classes. See
    `nlnas.correction.clustering.class_otm_matching`.
    """

    targets: dict[int, Tensor]
    """
    Dict that maps a true class to a correctly clustered sample in that true
    class. Note that not every true class may be represented in this dict.
    """

    y_clst: np.ndarray
    """`(N,)` vector of cluster labels."""


class BaseClassifier(pl.LightningModule):
    """
    Base image classifier class that supports LCC.

    Warning:
        When subclassing this, remember that the forward method must be able to
        deal with either `Tensor` or `Batch` inputs, and must return a logit
        `Tensor`.
    """

    _lcc_data: dict[str, LatentClusteringData] | None = None
    """
    If LCC is applied, then this is non `None` and updated at the begining of
    each epoch. See also `full_dataset_latent_clustering`.
    """

    standard_loss: nn.Module
    """'Standard' loss to use together with LCC."""

    def __init__(
        self,
        n_classes: int,
        lcc_submodules: list[str] | None = None,
        lcc_kwargs: dict | None = None,
        ce_weight: float = 1,
        image_key: Any = 0,
        label_key: Any = 1,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            n_classes (int):
            lcc_submodules (list[str] | None, optional): Submodules to consider
                for the latent correction loss. If `None` or `[]`, LCC is not
                performed
            lcc_kwargs (dict | None, optional): Optional parameters for LCC.
                Expected entries (all optional) are:
                * **weight (float):** Defaults to $10^{-4}$
                * **class_selection
                    (`nlnas.correction.LCCClassSelection` | None):** Defaults to
                    `None`, which means all classes are considered for
                    correction
                * **interval (int):** Apply LCC every `interval` epochs.
                    Defaults to $1$, meaning LCC will be applied every epoch
                    (after warmup).
                * **warmup (int):** Number of epochs to wait before
                    starting LCC. Defaults to $0$, meaning LCC will start
                    immediately.
                * **k (int):** Number of nearest neighbors to consider for LCC,
                  and Louvain clustering.
                * **pca_dim (int):** Samples are reduced to this dimension
                  before constructing the KNN graph. This must be at most the
                  batch size.
            ce_weight (float, optional): Weight of the cross-entropy loss in the
                clustering-CE loss. Ignored if LCC is not applied. Defaults to
                $1$.
            image_key (Any, optional): A batch passed to the model can be a
                tuple (most common) or a dict. This parameter specifies the key
                to use to retrieve the input tensor.
            label_key (Any, optional): Analogous to `image_key`.
            kwargs: Forwarded to
                [`pl.LightningModule`](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#)
        """
        super().__init__(**kwargs)
        self.save_hyperparameters(
            "ce_weight",
            "image_key",
            "label_key",
            "lcc_kwargs",
            "lcc_submodules",
            "n_classes",
        )
        if lcc_submodules:
            validate_lcc_kwargs(lcc_kwargs)
        self.standard_loss = BinaryCrossEntropy()

    def _evaluate(self, batch: Batch, stage: str | None = None) -> Tensor:
        """Self-explanatory"""
        image_key = self.hparams["image_key"]
        label_key = self.hparams["label_key"]
        x, y = batch[image_key], batch[label_key].to(self.device)
        latent: dict[str, Tensor] = {}
        logits = self.forward_intermediate(
            x, self.lcc_submodules, latent, keep_gradients=True
        )
        assert isinstance(logits, Tensor)
        loss_ce = self.standard_loss(logits, y)
        if self._lcc_data and stage == "train":
            idx = to_array(batch["_idx"])
            _losses = [
                lcc_loss(
                    z=z,
                    y_true=y,
                    y_clst=self._lcc_data[sm].y_clst[idx],
                    matching=self._lcc_data[sm].matching,
                    targets=self._lcc_data[sm].targets,
                    n_true_classes=self.hparams["n_classes"],
                )
                for sm, z in latent.items()
            ]
            loss_lcc = (
                torch.stack(_losses).mean()
                if _losses
                else torch.tensor(0.0, requires_grad=True)
                # ↑ actually need grad?
            )
            lcc_weight = self.hparams.get("lcc_kwargs", {}).get("weight", 1e-4)
            loss = self.hparams["ce_weight"] * loss_ce + lcc_weight * loss_lcc
            self.log(f"{stage}/lcc", loss_lcc, sync_dist=True)
        else:
            loss = loss_ce
        if stage:
            self.log_dict(
                {f"{stage}/loss": loss, f"{stage}/ce": loss_ce},
                sync_dist=True,
            )
            self.log(
                f"{stage}/acc",
                multiclass_accuracy(
                    logits,
                    y,
                    num_classes=self.hparams["n_classes"],
                    average="micro",
                ),
                prog_bar=True,
                sync_dist=True,
            )
        return loss  # type: ignore

    def configure_optimizers(self) -> Any:
        optimizer = create_optimizer_v2(self.parameters(), opt="lamb", lr=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/loss",
            },
        }

    def forward_intermediate(
        self,
        inputs: Tensor | Batch | list[Tensor] | Sequence[Batch],
        submodules: list[str],
        output_dict: dict,
        keep_gradients: bool = False,
    ) -> Tensor | list[Tensor]:
        """
        Runs the model and collects the output of specified submodules. The
        intermediate outputs are stored in `output_dict` under the
        corresponding submodule name. In particular, this method has side
        effects.

        Args:
            x (Tensor | Batch | list[Tensor] | list[Batch]): If batched (i.e.
                `x` is a list), then so is the output of this function and the
                entries in the `output_dict`
            submodules (list[str]):
            output_dict (dict):
            keep_gradients (bool, optional): If `True`, the tensors in
                `output_dict` keep their gradients (if they had some on the
                first place). If `False`, they are detached and moved to the
                CPU.
        """

        def maybe_detach(x: Tensor) -> Tensor:
            return x if keep_gradients else x.detach().cpu()

        def create_hook(key: str) -> Callable[[nn.Module, Any, Any], None]:
            def hook(_model: nn.Module, _args: Any, out: Any) -> None:
                if (
                    isinstance(out, (list, tuple))
                    and len(out) == 1
                    and isinstance(out[0], Tensor)
                ):
                    out = out[0]
                elif (  # Special case for ViTs
                    isinstance(out, (list, tuple))
                    and len(out) == 2
                    and isinstance(out[0], Tensor)
                    and not isinstance(out[1], Tensor)
                ):
                    out = out[0]
                elif not isinstance(out, Tensor):
                    raise ValueError(
                        f"Unsupported latent object type: {type(out)}: {out}."
                    )
                if batched:
                    if key not in output_dict:
                        output_dict[key] = []
                    output_dict[key].append(maybe_detach(out))
                else:
                    output_dict[key] = maybe_detach(out)

            return hook

        batched = isinstance(inputs, (list, tuple))
        handles: list[RemovableHandle] = []
        for name in submodules:
            submodule = self.get_submodule(name)
            handles.append(submodule.register_forward_hook(create_hook(name)))
        if batched:
            logits = [
                maybe_detach(
                    self.forward(
                        batch
                        if isinstance(batch, Tensor)
                        else batch[self.hparams["image_key"]]
                    )
                )
                for batch in inputs
            ]
        else:
            logits = maybe_detach(  # type: ignore
                self.forward(
                    inputs
                    if isinstance(inputs, Tensor)
                    else inputs[self.hparams["image_key"]]
                )
            )
        for h in handles:
            h.remove()
        return logits

    @staticmethod
    def get_image_processor(model_name: str, **kwargs: Any) -> Callable:
        """
        Returns an image processor for the model. By defaults, returns the
        identity function.
        """
        return lambda input: input

    @property
    def lcc_submodules(self) -> list[str]:
        """
        Returns the list of submodules considered for LCC, whith correct prefix
        if needed.

        TODO:
            Move to :class:`nlnas.classifier.WrappedClassifier` instead.
        """
        return (
            []
            if not self.hparams["lcc_submodules"]
            else [
                (sm if sm.startswith("model.") else "model." + sm)
                for sm in self.hparams["lcc_submodules"]
            ]
        )

    def on_train_batch_end(self, *args: Any, **kwargs: Any) -> None:
        """Just logs all optimizer's learning rate"""
        log_optimizers_lr(self, sync_dist=True)
        super().on_train_batch_end(*args, **kwargs)

    def on_train_epoch_end(self) -> None:
        """Cleans up training specific temporary attributes"""
        self._lcc_data = None
        super().on_train_epoch_end()

    def on_train_epoch_start(self) -> None:
        """
        Performs dataset-wide latent clustering and stores the results in
        private attribute `BaseClassifier._lc_data`.
        """
        # wether to apply LCC this epoch
        lcc_kwargs = self.hparams.get("lcc_kwargs") or {}
        do_lcc = (
            # we are passed warmup (lcc_warmup being None is equivalent to no
            # warmup)...
            self.current_epoch >= (lcc_kwargs.get("warmup") or 0)
            and (
                # ... and an LCC interval is specified...
                lcc_kwargs.get("interval") is not None
                # ... and the current epoch can have LCC done...
                and self.current_epoch % int(lcc_kwargs.get("interval", 1))
                == 0
            )
            # ... and there are submodule selected for LCC...
            and self.lcc_submodules
            # ... and the LCC weight is non-zero
            and lcc_kwargs.get("weight", 0) > 0
        )
        if do_lcc:
            # The import has to be here to prevent circular imports while having
            # all the FDLC logic in a separate file
            from .fdlc import full_dataset_latent_clustering

            with temporary_directory(self) as tmp_path:
                self._lcc_data = full_dataset_latent_clustering(
                    model=self,
                    output_dir=tmp_path,
                    # classes=lcc_kwargs.get("class_selection"),
                    tqdm_style="console",
                )
        super().on_train_epoch_start()

    def on_train_start(self) -> None:
        """
        Explicitly registers hyperparameters and metrics. You'd think Lightning
        would do this automatically, but nope.
        """
        self.logger.log_hyperparams(  # type: ignore
            self.hparams,
            {
                s + "/" + m: np.nan
                for s, m in product(
                    ["train", "val"], ["acc", "loss", "ce", "lcc"]
                )
            },
        )
        super().on_train_start()

    def test_step(self, batch: Batch, *_: Any, **__: Any) -> Tensor:
        """Override from `pl.LightningModule.test_step`."""
        return self._evaluate(batch, "test")

    def training_step(self, batch: Batch, *_: Any, **__: Any) -> Tensor:
        """Override from `pl.LightningModule.training_step`."""
        return self._evaluate(batch, "train")

    def validation_step(self, batch: Batch, *_: Any, **__: Any) -> Tensor:
        """Override from `pl.LightningModule.validation_step`."""
        return self._evaluate(batch, "val")
