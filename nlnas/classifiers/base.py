"""Base image classifier class that support clustering correction loss"""

from typing import Any, TypeAlias

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle
from torchmetrics.functional.classification import multiclass_accuracy

from ..correction import louvain_loss

Batch: TypeAlias = dict[Any, Tensor] | list[Tensor] | tuple[Tensor, ...]

OPTIMIZERS: dict[str, type] = {
    "asgd": torch.optim.ASGD,
    "adadelta": torch.optim.Adadelta,
    "adagrad": torch.optim.Adagrad,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "adamax": torch.optim.Adamax,
    "lbfgs": torch.optim.LBFGS,
    "nadam": torch.optim.NAdam,
    "optimizer": torch.optim.Optimizer,
    "radam": torch.optim.RAdam,
    "rmsprop": torch.optim.RMSprop,
    "rprop": torch.optim.Rprop,
    "sgd": torch.optim.SGD,
    "sparseadam": torch.optim.SparseAdam,
}

SCHEDULERS: dict[str, type] = {
    "constantlr": torch.optim.lr_scheduler.ConstantLR,
    "cosineannealinglr": torch.optim.lr_scheduler.CosineAnnealingLR,
    "cosineannealingwarmrestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    "cycliclr": torch.optim.lr_scheduler.CyclicLR,
    "exponentiallr": torch.optim.lr_scheduler.ExponentialLR,
    "lambdalr": torch.optim.lr_scheduler.LambdaLR,
    "linearlr": torch.optim.lr_scheduler.LinearLR,
    "multisteplr": torch.optim.lr_scheduler.MultiStepLR,
    "multiplicativelr": torch.optim.lr_scheduler.MultiplicativeLR,
    "onecyclelr": torch.optim.lr_scheduler.OneCycleLR,
    "polynomiallr": torch.optim.lr_scheduler.PolynomialLR,
    "reducelronplateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "sequentiallr": torch.optim.lr_scheduler.SequentialLR,
    "steplr": torch.optim.lr_scheduler.StepLR,
}


class BaseClassifier(pl.LightningModule):
    """
    See module documentation

    Warning:
        When subclassing this, remember that the forward method must be able to
        deal with either `Tensor` or `Batch` inputs, and must return a logit
        `Tensor`.
    """

    n_classes: int

    cor_submodules: list[str]
    cor_weight: float
    cor_kwargs: dict[str, Any]

    image_key: Any
    label_key: Any
    logit_key: Any

    # pylint: disable=unused-argument
    def __init__(
        self,
        n_classes: int,
        cor_submodules: list[str] | None = None,
        cor_weight: float = 1e-1,
        cor_kwargs: dict[str, Any] | None = None,
        image_key: Any = 0,
        label_key: Any = 1,
        logit_key: Any = None,
        optimizer: str = "sgd",
        optimizer_kwargs: dict[str, Any] | None = None,
        scheduler: str | None = None,
        scheduler_kwargs: dict[str, Any] | None = None,
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
            image_key (Any, optional): A batch passed to the model can be a
                tuple (most common) or a dict. This parameter specifies the key
                to use to retrieve the input tensor.
            label_key (Any, optional): Analogous to `image_key`
            logit_key (Any, optional): Analogous to `image_key` and `label_key`
                but used to extract the logits from the model's output. Leave
                to `None` if the model already returns logit tensors. If
                `model`is a Hugging Face transformer that outputs a
                [`ImageClassifierOutput`](https://huggingface.co/docs/transformers/v4.39.3/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput)
                or a
                [`ImageClassifierOutputWithNoAttention`](https://huggingface.co/docs/transformers/v4.39.3/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput),
                then this key should be
                `"logits"`.
            optimizer (str, optional): Optimizer name, case insensitive. See
                `OPTIMIZERS` and
                https://pytorch.org/docs/stable/optim.html#algorithms .
            optimizer_kwargs (dict, optional): Forwarded to the optimizer
                constructor
            scheduler (str, optional): Scheduler name, case insensitive. See
                `SCHEDULERS`. If left to `None`, then no scheduler is used.
            scheduler_kwargs (dict, optional): Forwarded to the scheduler, if
                any.
            kwargs: Forwarded to
                [`pl.LightningModule`](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#)
        """
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["model"])
        self.n_classes = n_classes
        self.cor_submodules = cor_submodules or []
        self.cor_weight, self.cor_kwargs = cor_weight, cor_kwargs or {}
        self.image_key, self.label_key = image_key, label_key
        self.logit_key = logit_key

    def _evaluate(self, batch: Batch, stage: str | None = None) -> Tensor:
        """Self-explanatory"""
        x, y = batch[self.image_key], batch[self.label_key].to(self.device)
        latent: dict[str, Tensor] = {}
        logits = self.forward_intermediate(
            x, self.cor_submodules, latent, keep_gradients=True
        )
        loss_ce = nn.functional.cross_entropy(logits, y.long())
        compute_correction_loss = (
            stage == "train" and self.cor_submodules and self.cor_weight > 0
        )
        if compute_correction_loss:
            loss_lou = torch.stack(
                [
                    louvain_loss(v, y, **self.cor_kwargs)
                    for v in latent.values()
                ]
            ).mean()
        else:
            loss_lou = torch.tensor(0.0)
        loss = loss_ce + self.cor_weight * loss_lou
        if stage:
            log = {
                f"{stage}/loss": loss,
                f"{stage}/ce": loss_ce,
                f"{stage}/acc": multiclass_accuracy(
                    logits, y, num_classes=self.n_classes, average="micro"
                ),
            }
            if compute_correction_loss:
                log[f"{stage}/louvain"] = loss_lou
            self.log_dict(log, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        cls = OPTIMIZERS[self.hparams["optimizer"].lower()]
        optimizer = cls(
            self.parameters(),
            **(self.hparams.get("optimizer_kwargs") or {}),
        )
        if self.hparams["scheduler"]:
            cls = SCHEDULERS[self.hparams["scheduler"]]
            scheduler = cls(
                optimizer,
                **(self.hparams.get("scheduler_kwargs") or {}),
            )
            return {"optimizer": optimizer, "scheduler": scheduler}
        return optimizer

    def forward_intermediate(
        self,
        inputs: Tensor | Batch,
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
            x (Tensor | Batch):
            submodules (list[str]):
            output_dict (dict):
            keep_gradients (bool, optional): If `True`, the tensors in
                `output_dict` keep their gradients (if they had some on the
                first place). If `False`, they are detached and moved to the
                CPU.
        """

        def create_hook(key: str):
            def hook(_model: nn.Module, _args: Any, out: Any) -> None:
                if isinstance(out, (tuple, list)) and len(out) == 1:
                    # This scenario happens with some implementations of ViTs
                    out = out[0]
                if isinstance(out, Tensor):
                    output_dict[key] = (
                        out if keep_gradients else out.cpu().detach()
                    )
                else:
                    raise ValueError(
                        f"Unsupported latent object type: {type(out)}. "
                        "Supported types are Tensors or tuples/lists "
                        "containing a single Tensor."
                    )

            return hook

        handles: list[RemovableHandle] = []
        for name in submodules:
            submodule = self.get_submodule(name)
            handles.append(submodule.register_forward_hook(create_hook(name)))
        x = inputs if isinstance(inputs, Tensor) else inputs[self.image_key]
        logits = self.forward(x)
        for h in handles:
            h.remove()
        return logits

    def on_train_batch_end(self, *args, **kwargs) -> None:
        def _lr(o: torch.optim.Optimizer) -> float:
            return o.param_groups[0]["lr"]

        opts = self.optimizers()
        if isinstance(opts, list):
            self.log_dict(
                {
                    f"lr_{i}": _lr(opt)
                    for i, opt in enumerate(opts)
                    if isinstance(opt, torch.optim.Optimizer)
                }
            )
        elif isinstance(opts, torch.optim.Optimizer):
            self.log("lr", _lr(opts))
        return super().on_train_batch_end(*args, **kwargs)

    # pylint: disable=arguments-differ
    def test_step(self, batch: Batch, *_, **__) -> Tensor:
        return self._evaluate(batch, "test")

    # pylint: disable=arguments-differ
    def training_step(self, batch: Batch, *_, **__) -> Tensor:
        return self._evaluate(batch, "train")

    def validation_step(self, batch: Batch, *_, **__) -> Tensor:
        return self._evaluate(batch, "val")
