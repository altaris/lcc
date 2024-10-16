"""General utility functions for classifiers."""

from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor

from ..correction import LCC_CLASS_SELECTIONS
from ..utils import to_array

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


def inflate_vector(
    v: np.ndarray | Tensor | list[float],
    mask: np.ndarray | Tensor | list[bool],
) -> np.ndarray:
    """
    Say `v` has shape (n_a,) while `mask` has shape (n_b,). This function
    "inflates" `v` into a vector `w` of shape (n_b,) such that `v = w[mask]`.
    Values of `w` that don't fall in the mask are set to -1.
    """
    if to_array(mask).all():
        return to_array(v)
    v, mask = to_array(v), to_array(mask).astype(bool)
    w = np.full_like(mask, -1, dtype=v.dtype)
    w[mask] = v
    return w


def log_optimizers_lr(model: pl.LightningModule, **kwargs: Any) -> None:
    """
    Logs all optimizers learning rates to TensorBoard.

    Args:
        model (pl.LightningModule):
        kwargs: Passed to
            [`pl.LightningModule.log_dict`](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#log-dict)
    """
    _lr = lambda opt: opt.param_groups[0]["lr"]
    optimizers = model.optimizers()
    if isinstance(optimizers, list):
        model.log_dict(
            {
                f"lr_{i}": _lr(opt)
                for i, opt in enumerate(optimizers)
                if isinstance(opt, torch.optim.Optimizer)
            },
            **kwargs,
        )
    elif isinstance(optimizers, torch.optim.Optimizer):
        model.log("lr", _lr(optimizers), **kwargs)


def validate_lcc_kwargs(lcc_kwargs: dict[str, Any] | None) -> None:
    """
    Makes sure that an LCC hyperparameter dict is valid. Used in constructors of
    LCC enabled classifiers.

    Args:
        lcc_kwargs (dict[str, Any] | None):
    """
    if not lcc_kwargs:
        return
    if (x := lcc_kwargs.get("weight", 1)) <= 0:
        raise ValueError(f"LCC weight must be positive, got {x}")
    if (x := lcc_kwargs.get("interval", 1)) < 1:
        raise ValueError(f"LCC interval must be at least 1, got {x}")
    if (x := lcc_kwargs.get("warmup", 0)) < 0:
        raise ValueError(f"LCC warmup must be at least 0, got {x}")
    if (x := lcc_kwargs.get("class_selection")) not in LCC_CLASS_SELECTIONS + [
        None
    ]:
        raise ValueError(
            f"Invalid class selection policy '{x}'. Available: policies are: "
            + ", ".join(map(lambda a: f"`{a}`", LCC_CLASS_SELECTIONS))
            + ", or `None`"
        )
