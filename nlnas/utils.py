"""General utilities"""

import os
import re
from contextlib import contextmanager
from glob import glob
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from loguru import logger as logging
from safetensors import numpy as nst
from pytorch_lightning.strategies.strategy import Strategy


class NoCheckpointFound(Exception):
    """Raised by `nlnas.utils.last_checkpoint_path` if no checkpoint is found"""


def best_device() -> str:
    """Self-explanatory"""
    accelerator = os.getenv("PL_ACCELERATOR", "auto").lower()
    if accelerator == "gpu" and torch.cuda.is_available():
        return "cuda"
    if accelerator == "auto":
        return (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    return accelerator


def last_checkpoint_path(checkpoints_dir_path: Path) -> Path:
    """
    Finds the file path of the last Pytorch Lightning training checkpoint
    (`ckpt` file) in a given directory. The step count is considered, rather
    than the epoch count.
    """
    d, r = {}, r".*step=(\d+)\.ckpt"
    for c in glob(str(checkpoints_dir_path / "*step=*.ckpt")):
        m = re.match(r, c)
        if m:
            d[int(m.group(1))] = c
    ss = list(d.keys())
    if not ss:
        raise NoCheckpointFound
    sm = max(ss)
    return Path(d[sm])


def pl_module_loader(
    cls: type, root_dir: str | Path, name: str, version: int = 0
) -> pl.LightningModule:
    """
    Loader for pytorch lightning modules, to be used with
    `nlnas.utils.produces_artifact`.
    """
    assert issubclass(cls, pl.LightningModule)
    if not isinstance(root_dir, Path):
        root_dir = Path(root_dir)
    ckpt = last_checkpoint_path(
        root_dir / "tb_logs" / name / f"version_{version}" / "checkpoints"
    )
    logging.debug("Loading checkpoint '{}'", ckpt)
    module: pl.LightningModule = cls.load_from_checkpoint(str(ckpt))  # type: ignore
    return module


def produces_artifact(
    loader: Callable,
    saver: Callable | None = None,
    loader_args: Any = None,
    loader_kwargs: dict | None = None,
    saver_args: Any = None,
    saver_kwargs: dict | None = None,
) -> Callable:
    """
    Calls the loader function and returns the result. If the loader throws an
    exception, runs the decorated function instead. If a saver method is given,
    it is run on the results as

    ```py
    saver(results, *saver_args, **saver_kwargs)
    ```

    Finally, the results are returned. Here's an example to guard a model
    training:

    ```py
    _train = produces_artifact(
        pl_module_loader,
        loader_kwargs={
            "cls": type(model),
            "root_dir": output_dir,
            "name": "my_model",
        },
    )(train_model)
    model = _train(model, train, val, root_dir=output_dir)
    ```

    The signature of `_train` is the same as `nlnas.utils.train_model`.

    """

    loader_args = loader_args or []
    loader_kwargs = loader_kwargs or {}
    saver_args = saver_args or []
    saver_kwargs = saver_kwargs or {}

    def _decorator(function: Callable) -> Callable:
        def _wrapped(*args, **kwargs) -> Any:
            try:
                data = loader(*loader_args, **loader_kwargs)  # type: ignore
                logging.debug(
                    "Skipped call to guarded method '{}'", function.__name__
                )
                return data
            except TypeError:
                raise
            except:
                results = function(*args, **kwargs)
                if saver is not None:
                    saver(results, *saver_args, **saver_kwargs)  # type: ignore
                return results

        return _wrapped

    return _decorator


def produces_numpy_array(path: str | Path):
    """
    Convenience function to use instead of `nlnas.utils.produces_artifact` when
    the artifact is a *single* numpy array. Here's an example of how to guard a
    function `f` that returns a numpy array:

    ```py
    _f = produces_numpy_array(output_dir / "data.nst")(f)
    arr = _f(*args, **kwargs)
    ```

    Under the hood, the returned array `arr` is wrapped in the document
    `{"data": arr}` and saved using the numpy API of
    [`safetensors`](https://huggingface.co/docs/safetensors/api/numpy).

    """

    def _load(p: Path):
        return nst.load_file(p)["data"]

    def _save(arr: np.ndarray, p: Path):
        nst.save_file({"data": arr}, p)

    if not isinstance(path, Path):
        path = Path(path)
    return produces_artifact(
        loader=_load,
        loader_args=[path],
        saver=_save,
        saver_args=[path],
    )


def train_model(
    model: pl.LightningModule,
    train_dl: DataLoader,
    val_dl: DataLoader,
    root_dir: str | Path,
    name: str | None = None,
    max_epochs: int = 512,
    additional_callbacks: list[pl.Callback] | None = None,
    early_stopping_kwargs: dict | None = None,
    strategy: str | Strategy = "ddp",
    **kwargs,
) -> pl.LightningModule:
    """
    Convenience function to invoke Pytorch Lightning's trainer fit. Returns the
    best checkpoint. The accelerator is set to `auto` unless otherwise directed
    by the `PL_ACCELERATOR` environment variable. If the accelerator ends up
    using a CUDA `gpu`, the trainer uses a
    [`DDPStrategy`](https://pytorch-lightning.readthedocs.io/en/latest/api/lightning.pytorch.strategies.DDPStrategy.html).

    Args:
        model (pl.LightningModule): The model to train. In its
            `validation_step`, the model must log the `val/loss` metric.
        train_dl (DataLoader): The train dataloader.
        val_dl (DataLoader): The validation dataloader.
        root_dir (str | Path): The root dir of the trainer. The
            tensorboard logs will be stored under `root_dir/tb_logs/name` and
            the CSV logs under `root_dir/csv_logs/name`.
        name (str, optional): The name of the model. The
            tensorboard logs will be stored under `root_dir/tb_logs/name`.
        max_epochs (int): The maximum number of epochs. Note that an early
            stopping callbacks with a patience of 10 monitors the `val/loss`
            metric by default.
        additional_callbacks (list[pl.Callback], optional): Additional
            trainer callbacks. Note that the following callbacks are
            automatically set:
            ```py
            pl.callbacks.EarlyStopping(monitor="val/loss", patience=10),
            pl.callbacks.LearningRateMonitor("epoch"),
            pl.callbacks.ModelCheckpoint(save_weights_only=True),
            pl.callbacks.RichProgressBar(),
            ```
        early_stopping_kwargs (dict, optional): kwargs for the [`pl.callbacks.EarlyStopping`](https://pytorch-lightning.readthedocs.io/en/latest/api/lightning.pytorch.callbacks.EarlyStopping.html)
            callback. By default, it is
            ```py
            {
                monitor="val/loss",
                patience=10,
            }
            ```
        strategy (Union[str, Strategy]): Strategy to use (duh). Defaults to
            `ddp`, but if running in a notebook, use `dp` or `ddp_notebook`.
        **kwargs: Forwarded to the [`pl.Trainer`
            constructor](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#init).
    """

    if not isinstance(root_dir, Path):
        root_dir = Path(root_dir)

    name = name or model.__class__.__name__.lower()
    logging.info("Training model '{}' in '{}'", name, root_dir)

    accelerator = os.getenv("PL_ACCELERATOR", "auto").lower()
    logging.debug("Set accelerator to '{}'", accelerator)
    if accelerator in ["auto", "gpu"] and torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")  # hehe matmul go brrrr

    # https://stackoverflow.com/questions/48250053/pytorchs-dataloader-too-many-open-files-error-when-no-files-should-be-open
    # https://github.com/pytorch/pytorch/issues/11201
    torch.multiprocessing.set_sharing_strategy("file_system")

    if not additional_callbacks:
        additional_callbacks = []
    if not early_stopping_kwargs:
        early_stopping_kwargs = {
            "monitor": "val/loss",
            "patience": 10,
            "mode": "min",
        }

    tb_logger = pl.loggers.TensorBoardLogger(
        str(root_dir / "tb_logs"),
        name=name,
        default_hp_metric=False,
        log_graph=False,
    )
    csv_logger = pl.loggers.CSVLogger(
        str(root_dir / "csv_logs"),
        name=name,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[
            pl.callbacks.EarlyStopping(**early_stopping_kwargs),
            # pl.callbacks.LearningRateMonitor("epoch"),
            pl.callbacks.ModelCheckpoint(
                save_top_k=-1,
                monitor="val/loss",
                mode="min",
                every_n_epochs=1,
            ),
            pl.callbacks.RichProgressBar(),
            # pl.callbacks.BatchSizeFinder(),
            *additional_callbacks,
        ],
        default_root_dir=str(root_dir),
        logger=[tb_logger, csv_logger],
        accelerator=accelerator,
        strategy=strategy,
        **kwargs,
    )

    trainer.fit(model, train_dl, val_dl)

    assert trainer.checkpoint_callback is not None
    ckpt = Path(trainer.checkpoint_callback.best_model_path)
    logging.debug("Loading best checkpoint '{}'", ckpt)
    return type(model).load_from_checkpoint(str(ckpt))  # type: ignore


def train_model_guarded(
    model: pl.LightningModule,
    train_dl: DataLoader,
    val_dl: DataLoader,
    root_dir: str | Path,
    name: str,
    *args,
    **kwargs,
) -> pl.LightningModule:
    """
    Guarded version of `nlnas.utils.train_model`, i.e. if a checkpoint already
    exists for the model, it is loaded and returned instead of training the
    model.

    See also:
        `nlnas.utils.produces_artifact`, `nlnas.utils.pl_module_loader`
    """
    _train = produces_artifact(
        pl_module_loader,
        loader_kwargs={
            "cls": type(model),
            "root_dir": root_dir,
            "name": name,
        },
    )(train_model)
    return _train(model, train_dl, val_dl, root_dir, name, *args, **kwargs)
