"""General utility functions for classifiers."""

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator

import numpy as np
import pytorch_lightning as pl
import torch
from numpy.typing import ArrayLike

from ..correction import LCC_CLASS_SELECTIONS
from ..utils import to_array


def inflate_vector(v: ArrayLike, mask: ArrayLike) -> np.ndarray:
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


@contextmanager
def temporary_directory(model: pl.LightningModule) -> Iterator[Path]:
    """
    Makes rank 0 create a temporary directory and propagates the path to all
    other ranks. This context manager also acts as a barrier (due to the
    directory name being broadcasted), so there is no need for explicit
    syncronization on top of this.
    """
    if model.trainer.global_rank == 0:
        handler = TemporaryDirectory(prefix="lcc-")
        name = handler.name
    else:
        name = None
    name = model.trainer.strategy.broadcast(name, src=0)
    assert isinstance(name, str)  # for typechecking
    try:
        yield Path(name)
    finally:
        model.trainer.strategy.barrier()  # Make sure everyone is done with tmp
        if model.trainer.global_rank == 0:
            handler.cleanup()


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
