"""General model training utilities"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd
import pytorch_lightning as pl
import regex as re
import torch
import turbo_broccoli as tb
from loguru import logger as logging

from .classifiers import BaseClassifier, HuggingFaceClassifier, TimmClassifier
from .datasets import HuggingFaceDataset
from .logging import r0_debug, r0_info

DEFAULT_MAX_GRAD_NORM = 1.0


class NoCheckpointFound(Exception):
    """Raised by `nlnas.utils.last_checkpoint_path` if no checkpoint is found"""


def all_checkpoint_paths(output_path: str | Path) -> list[Path]:
    """
    Returns the sorted (by epoch) list of all checkpoints.

    Args:
        output_path (str | Path): e.g.
            `out.local/ft/cifar100/microsoft-resnet-18`

    Raises:
        NoCheckpointFound: If no checkpoint is found
    """
    r, d = re.compile(r"/epoch=(\d+)-step=\d+\.ckpt$"), {}
    for p in Path(output_path).glob("**/*.ckpt"):
        if m := re.search(r, str(p)):
            epoch = int(m.group(1))
            d[epoch] = p
    ckpts = [d[i] for i in sorted(list(d.keys()))]
    if not ckpts:
        raise NoCheckpointFound
    return ckpts


def best_checkpoint_path(
    output_path: str | Path,
    metric: str = "val/acc",
    mode: Literal["min", "max"] = "max",
) -> tuple[Path, int]:
    """
    Returns the path to the best checkpoint

    Args:
        output_path (str | Path): e.g.
            `out.local/ft/cifar100/microsoft-resnet-18`
        metric (str, optional):
        mode (Literal["min", "max"], optional):

    Returns:
        tuple[Path, int]: _description_
    """
    if not isinstance(output_path, Path):
        output_path = Path(output_path)
    ckpts = all_checkpoint_paths(output_path)
    metrics_path = list(output_path.glob("**/csv_logs/**/metrics.csv"))[0]
    epoch = best_epoch(metrics_path, metric, mode)
    return ckpts[epoch], epoch


def best_epoch(
    metrics_path: str | Path,
    metric: str = "val/acc",
    mode: Literal["min", "max"] = "max",
) -> int:
    """Given the `metrics.csv` path, returns the best epoch index"""
    df = pd.read_csv(metrics_path)
    df.drop(columns=["train/loss"], inplace=True)
    df = df.groupby("epoch").tail(1)
    df.reset_index(inplace=True, drop=True)
    return int(df[metric].argmax() if mode == "max" else df[metric].argmin())


def checkpoint_ves(path: str | Path) -> tuple[int, int, int]:
    """
    Given a checkpoint path that looks like e.g.

        out/resnet18/cifar10/model/tb_logs/resnet18/version_2/checkpoints/epoch=32-step=5181.ckpt

    returns the **v**ersion number (2), the number of **e**pochs (32), and the
    number of **s**teps (5181).
    """
    r = r".*version_(\d+)/checkpoints/epoch=(\d+)-step=(\d+).*\.ckpt"
    if m := re.match(r, str(path)):
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    raise ValueError(f"Path '{path}' is not a valid checkpoint path")


def last_checkpoint_path(output_path: Path) -> Path:
    """
    Finds the file path of the last Pytorch Lightning training checkpoint
    (`ckpt` file) in a given directory. The step count is considered, rather
    than the epoch count.

    Args:
        output_path (str | Path): e.g.
            `out.local/ft/cifar100/microsoft-resnet-18`
    """
    d = {}
    for c in output_path.glob("**/*step=*.ckpt"):
        try:
            d[checkpoint_ves(c)[2]] = c
        except ValueError:
            pass
    if ks := list(d.keys()):
        return Path(d[max(ks)])
    raise NoCheckpointFound


def make_trainer(
    model_name: str,
    output_dir: Path,
    max_epochs: int = 512,
    accelerator: str = "auto",
) -> pl.Trainer:
    """
    Self-explanatory

    Args:
        model_name (str):
        output_dir (Path):
        max_epochs (int, optional):
    """
    tb_logger = pl.loggers.TensorBoardLogger(
        str(output_dir / "tb_logs"), name=model_name, default_hp_metric=False
    )
    csv_logger = pl.loggers.CSVLogger(
        str(output_dir / "csv_logs"), name=model_name
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="val/loss", patience=20, mode="min"
            ),
            pl.callbacks.ModelCheckpoint(
                save_top_k=-1, monitor="val/loss", mode="min", every_n_epochs=1
            ),
            pl.callbacks.TQDMProgressBar(),
        ],
        default_root_dir=str(output_dir),
        logger=[tb_logger, csv_logger],
        log_every_n_steps=1,
        gradient_clip_val=DEFAULT_MAX_GRAD_NORM,
        accelerator=accelerator,
    )
    return trainer


def train(
    model_name: str,
    dataset_name: str,
    output_dir: Path,
    ckpt_path: Path | None = None,
    ce_weight: float = 1,
    lcc_submodules: list[str] | None = None,
    lcc_weight: float | None = None,
    lcc_interval: int | None = None,
    lcc_warmup: int | None = None,
    max_epochs: int = 100,
    batch_size: int = 64,
    train_split: str = "train",
    val_split: str = "val",
    test_split: str = "test",
    image_key: str = "image",
    label_key: str = "label",
    logit_key: str = "logits",
    head_name: str | None = None,
):
    """
    Performs fine-tuning on a model, possibly with latent clustering correction.

    Args:
        model_name (str):
        dataset_name (str):
        output_dir (Path):
        ckpt_path (Path | None): If `None`, the correction will start from the
            weights available on the Hugging Face model hub.
        ce_weight (float, optional):
        lcc_submodules (list[str]):
        lcc_weight (float, optional):
        lcc_interval (int, optional):
        lcc_warmup (int, optional):
        max_epochs (int, optional):
        batch_size (int, optional):
        train_split (str, optional):
        val_split (str, optional):
        test_split (str, optional):
        image_key (str, optional):
        label_key (str, optional):
        logit_key (str, optional):
        head_name (str | None, optional):
    """
    do_lcc = (
        (lcc_weight or 0) > 0 and (lcc_interval or 0) > 0 and lcc_submodules
    )
    if do_lcc:
        logging.debug("Performing latent cluster correction")
    if do_lcc and (n_cuda_devices := torch.cuda.device_count()) > 1:
        logging.critical(
            "Latent cluster correction only support CPU or single GPU "
            f"training , but found {n_cuda_devices} CUDA devices. Please set "
            "CUDA_VISIBLE_DEVICES to a single device, for example: "
            "export CUDA_VISIBLE_DEVICES=0"
        )
        sys.exit(1)

    torch.multiprocessing.set_sharing_strategy("file_system")

    _dataset_name = dataset_name.replace("/", "-")
    _model_name = model_name.replace("/", "-")
    _output_dir = output_dir / _dataset_name / _model_name
    _output_dir.mkdir(parents=True, exist_ok=True)

    classifier_cls: type[BaseClassifier]
    if model_name.startswith("timm/"):
        classifier_cls = TimmClassifier
    else:
        classifier_cls = HuggingFaceClassifier

    dataset = HuggingFaceDataset(
        dataset_name=dataset_name,
        fit_split=train_split,
        val_split=val_split,
        test_split=test_split,
        label_key=label_key,
        image_processor=classifier_cls.get_image_processor(model_name),
    )
    n_classes = dataset.n_classes()

    model = classifier_cls(
        model_name=model_name,
        n_classes=n_classes,
        head_name=head_name,
        image_key=image_key,
        label_key=label_key,
        logit_key=logit_key,
        optimizer="adam",
        optimizer_kwargs={"lr": 5e-5},
        lcc_weight=lcc_weight,
        lcc_submodules=lcc_submodules,
        lcc_class_selection=(
            "top_pair_5" if do_lcc else None
        ),  # TODO: make this configurable
        lcc_interval=lcc_interval,
        lcc_warmup=lcc_warmup,
        ce_weight=ce_weight,
    )
    if isinstance(ckpt_path, Path):
        # pylint: disable=no-value-for-parameter
        model.model = classifier_cls.load_from_checkpoint(  # type: ignore
            ckpt_path
        ).model
        r0_info("Loaded checkpoint '{}'", ckpt_path)
    r0_debug("Model hyperparameters:\n{}", json.dumps(model.hparams, indent=4))

    trainer = make_trainer(
        _model_name,
        _output_dir,
        max_epochs=max_epochs,
    )
    start = datetime.now()
    trainer.fit(model, dataset)
    fit_time = datetime.now() - start
    r0_info("Finished correction in {}", fit_time)

    ckpt = Path(trainer.checkpoint_callback.best_model_path)  # type: ignore
    ckpt = ckpt.relative_to(output_dir)
    v, e, s = checkpoint_ves(ckpt)
    r0_info("Best checkpoint path: '{}'", ckpt)
    r0_info("version={}, best_epoch={}, n_steps={}", v, e, s)

    test_results = trainer.test(model, dataset)

    document = {
        "dataset": {
            "name": dataset_name,
            "n_classes": n_classes,
            "train_split": train_split,
            "val_split": val_split,
            "test_split": test_split,
            "image_key": image_key,
            "label_key": label_key,
            "batch_size": batch_size,
        },
        "model": {
            "name": model_name,
            "hparams": dict(model.hparams),
            "max_epochs": max_epochs,
            "best_checkpoint": {
                "path": str(ckpt),
                "version": v,
                "best_epoch": e,
                "n_steps": s,
            },
            "time": fit_time / timedelta(seconds=1),
            "test": test_results,
        },
    }
    tb.save_json(document, _output_dir / f"results.{v}.json")
