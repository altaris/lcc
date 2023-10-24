"""General utilities"""

import os
from glob import glob
from pathlib import Path
from typing import Any, Callable, Literal, Tuple

import pandas as pd
import pytorch_lightning as pl
import regex as re
import torch
from loguru import logger as logging
from pytorch_lightning.strategies.strategy import Strategy


class NoCheckpointFound(Exception):
    """Raised by `nlnas.utils.last_checkpoint_path` if no checkpoint is found"""


def all_checkpoint_paths(ckpts_dir_path: str | Path) -> list[Path]:
    """
    Returns the sorted (by epoch) list of all checkpoint file paths in a given
    directory. `ckpts_dir_path` probably looks like
    `.../tb_logs/my_model/version_N/checkpoints`. The checkpoint files are
    assumed to be named as `epoch=XX-step=YY.ckpt` where of course `XX` is the
    epoch number and `YY` is the step number.

    Args:
        ckpts_dir_path (str | Path):
    """
    r, d = re.compile(r"/epoch=(\d+)-step=\d+\.ckpt$"), {}
    for p in glob(str(Path(ckpts_dir_path) / "*.ckpt")):
        if m := re.search(r, p):
            epoch = int(m.group(1))
            d[epoch] = Path(p)
    return [d[i] for i in sorted(list(d.keys()))]


def best_checkpoint_path(
    ckpts_dir_path: str | Path,
    metrics_csv_path: str | Path,
    metric: str = "val/acc",
    mode: Literal["min", "max"] = "max",
) -> Tuple[Path, int]:
    """Returns the path to the best checkpoint"""
    ckpts = all_checkpoint_paths(ckpts_dir_path)
    epoch = best_epoch(metrics_csv_path, metric, mode)
    return ckpts[epoch], epoch


def best_epoch(
    metrics_csv_path: str | Path,
    metric: str = "val/acc",
    mode: Literal["min", "max"] = "max",
) -> int:
    """Given the `metrics.csv` path, returns the best epoch index"""
    df = pd.read_csv(metrics_csv_path)
    df.drop(columns=["train/loss"], inplace=True)
    df = df.groupby("epoch").tail(1)
    df.reset_index(inplace=True, drop=True)
    return int(df[metric].argmax() if mode == "max" else df[metric].argmin())


def checkpoint_ves(path: str | Path) -> Tuple[int, int, int]:
    """
    Given a checkpoint path that looks like e.g.

        out/resnet18/cifar10/model/tb_logs/resnet18/version_2/checkpoints/epoch=32-step=5181.ckpt

    returns the version number (2), the number of epochs (32), and the number
    of steps (5181)
    """
    m = re.match(
        r".*version_(\d+)/checkpoints/epoch=(\d+)-step=(\d+)\.ckpt", str(path)
    )
    if m is None:
        raise ValueError(f"Path '{path}' is not a valid checkpoint path")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def last_checkpoint_path(checkpoints_dir_path: Path) -> Path:
    """
    Finds the file path of the last Pytorch Lightning training checkpoint
    (`ckpt` file) in a given directory. The step count is considered, rather
    than the epoch count.
    """
    d = {}
    for c in glob(str(checkpoints_dir_path / "*step=*.ckpt")):
        try:
            d[checkpoint_ves(c)[2]] = c
        except ValueError:
            pass
    ss = list(d.keys())
    if not ss:
        raise NoCheckpointFound
    sm = max(ss)
    return Path(d[sm])


def pl_module_loader(
    cls: type, root_dir: str | Path, name: str, version: int = 0
) -> Tuple[pl.LightningModule, Path]:
    """
    Loader for pytorch lightning modules, to be used with
    `nlnas.utils.produces_artifact`.
    """
    assert issubclass(cls, pl.LightningModule)  # For typechecking
    if not isinstance(root_dir, Path):
        root_dir = Path(root_dir)
    ckpt, _ = best_checkpoint_path(
        root_dir / "tb_logs" / name / f"version_{version}" / "checkpoints",
        root_dir / "csv_logs" / name / f"version_{version}" / "metrics.csv",
    )
    logging.debug("Loading checkpoint '{}'", ckpt)
    module: pl.LightningModule = cls.load_from_checkpoint(str(ckpt))  # type: ignore
    return module, ckpt


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


def train_model(
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    root_dir: str | Path,
    name: str | None = None,
    max_epochs: int = 512,
    additional_callbacks: list[pl.Callback] | None = None,
    early_stopping_kwargs: dict | None = None,
    strategy: str | Strategy = "ddp",
    reload: bool = False,
    **kwargs,
) -> Tuple[pl.LightningModule, Path]:
    """
    Convenience function to invoke Pytorch Lightning's trainer fit. Returns the
    best checkpoint. The accelerator is set to `auto` unless otherwise directed
    by the `PL_ACCELERATOR` environment variable. If the accelerator ends up
    using a CUDA `gpu`, the trainer uses a
    [`DDPStrategy`](https://pytorch-lightning.readthedocs.io/en/latest/api/lightning.pytorch.strategies.DDPStrategy.html).

    Args:
        model (pl.LightningModule): The model to train. In its
            `validation_step`, the model must log the `val/acc` metric.
        train_dl (DataLoader): The train dataloader.
        val_dl (DataLoader): The validation dataloader.
        root_dir (str | Path): The root dir of the trainer. The
            tensorboard logs will be stored under `root_dir/tb_logs/name` and
            the CSV logs under `root_dir/csv_logs/name`.
        name (str, optional): The name of the model. The
            tensorboard logs will be stored under `root_dir/tb_logs/name`.
        max_epochs (int): The maximum number of epochs. Note that an early
            stopping callbacks with a patience of 10 monitors the `val/acc`
            metric by default.
        additional_callbacks (list[pl.Callback], optional): Additional
            trainer callbacks. Note that the following callbacks are
            automatically set:
            ```py
            pl.callbacks.EarlyStopping(monitor="val/acc", patience=10),
            pl.callbacks.LearningRateMonitor("epoch"),
            pl.callbacks.ModelCheckpoint(save_weights_only=True),
            pl.callbacks.TQDMProgressBar(),
            ```
        early_stopping_kwargs (dict, optional): kwargs for the [`pl.callbacks.EarlyStopping`](https://pytorch-lightning.readthedocs.io/en/latest/api/lightning.pytorch.callbacks.EarlyStopping.html)
            callback. By default, it is
            ```py
            {
                monitor="val/acc",
                patience=10,
            }
            ```
        strategy (Union[str, Strategy]): Strategy to use (duh). Defaults to
            `ddp`, but if running in a notebook, use `dp` or `ddp_notebook`.
        reload (bool, optional): Wether to reload the best checkpoint after
            training
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

    additional_callbacks = additional_callbacks or []
    early_stopping_kwargs = early_stopping_kwargs or {
        "monitor": "val/acc",
        "patience": 10,
        "mode": "max",
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
            pl.callbacks.ModelCheckpoint(
                save_top_k=-1, monitor="val/acc", mode="max", every_n_epochs=1
            ),
            pl.callbacks.TQDMProgressBar(),
            *additional_callbacks,
        ],
        default_root_dir=str(root_dir),
        logger=[tb_logger, csv_logger],
        accelerator=accelerator,
        strategy=strategy,
        **kwargs,
    )

    trainer.fit(model, datamodule)

    ckpt = Path(trainer.checkpoint_callback.best_model_path)  # type: ignore
    v, e, s = checkpoint_ves(ckpt)
    logging.info(
        "Training completed: version={}, best_epoch={}, n_steps={}", v, e, s
    )
    if reload:
        logging.debug("Reloading best checkpoint '{}'", ckpt)
        model = type(model).load_from_checkpoint(str(ckpt))  # type: ignore
    return model, ckpt


def train_model_guarded(
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    root_dir: str | Path,
    name: str,
    *args,
    **kwargs,
) -> Tuple[pl.LightningModule, Path]:
    """
    Guarded version of `nlnas.utils.train_model`, i.e. if a checkpoint already
    exists for the model, it is loaded and returned instead of training the
    model. **WARNING** if checkpoints exist, only the most recent one is
    returned, which is not necessarily the best one.
    """
    _train = produces_artifact(
        pl_module_loader,
        loader_kwargs={
            "cls": type(model),
            "root_dir": root_dir,
            "name": name,
        },
    )(train_model)
    return _train(model, datamodule, root_dir, name, *args, **kwargs)
