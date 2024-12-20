"""General model training utilities"""

import hashlib
import json
import os
import uuid
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import pandas as pd
import pytorch_lightning as pl
import regex as re
import torch
import turbo_broccoli as tb

from lcc.classifiers.base import validate_lcc_kwargs

from .classifiers import get_classifier_cls
from .datasets import HuggingFaceDataset, get_dataset
from .logging import r0_debug, r0_info
from .utils import get_reasonable_n_jobs

# from .ema import EMACallback

DEFAULT_MAX_GRAD_NORM = 1.0
"""For gradient clipping."""


class NoCheckpointFound(Exception):
    """
    Raised by `lcc.training.all_checkpoint_paths` and
    `lcc.training.best_checkpoint_path` if no checkpoints are found.
    """


def _dict_sha1(d: dict) -> str:
    """
    Quick and dirty way to get a unique hash for a (potentially nested)
    dictionary.

    Warning:
        This method does not sort inner sets.
    """
    h = hashlib.sha1()
    h.update(json.dumps(d, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def all_checkpoint_paths(output_path: str | Path) -> list[Path]:
    """
    Returns the sorted (by epoch) list of all checkpoints. The checkpoint files
    must follow the following pattern:

        epoch=<digits>-step=<digits>.ckpt

    Args:
        output_path (str | Path): e.g.
            `out.local/ft/cifar100/microsoft-resnet-18`. There is no assumption
            on the structure of this folder, as long as it contains `.ckpt`
            files either directly or with subfolders in between.

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
    Returns the path to the best checkpoint.

    Args:
        output_path (str | Path): e.g.
            `out.local/ft/cifar100/microsoft-resnet-18`. This folder is expected
            to contain a `tb_logs` and `csv_logs` folder, either directly or
            with subfolders in between.
        metric (str, optional):
        mode (Literal["min", "max"], optional):

    Returns:
        A tuple containing the path to the checkpoint file, and the epoch
        number.
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


def checkpoint_ves(path: str | Path) -> tuple[str, int, int]:
    """
    Given a checkpoint path that looks like e.g.

        out/resnet18/cifar10/model/tb_logs/resnet18/060516dd86294076878cd278cfc59237/checkpoints/epoch=32-step=5181.ckpt

    returns the **v**ersion name (`060516dd86294076878cd278cfc59237`), the
    number of **e**pochs (32), and the number of **s**teps (5181).
    """
    r = r".*/(\w+)/checkpoints/epoch=(\d+)-step=(\d+).*\.ckpt"
    if m := re.match(r, str(path)):
        return str(m.group(1)), int(m.group(2)), int(m.group(3))
    raise ValueError(f"Path '{path}' is not a valid checkpoint path")


def make_trainer(
    output_dir: Path | str,
    model_name: str | None = None,
    max_epochs: int = 50,
    save_all_checkpoints: bool = False,
    stage: Literal["train", "test"] = "train",
    version: int | str | None = None,
) -> pl.Trainer:
    """
    Makes a [PyTorch Lightning
    `Trainer`](https://lightning.ai/docs/pytorch/stable/common/trainer.html)
    with some sensible defaults.

    Args:
        output_dir (Path | str):
        model_name (str): Ignored if `stage` is `test`, but must be set if
            `stage` is `train`.
        max_epochs (int, optional): Ignored if `stage` is `test`.
        save_all_checkpoints (bool, optional): If set to `False`, then only the
            best checkpoint is saved.
        stage (str, optional): Either `train` or `test`.
    """
    output_dir = Path(output_dir)

    config = {
        "default_root_dir": str(output_dir),
        "log_every_n_steps": 1,
    }
    if stage == "train":
        if model_name is None:
            raise ValueError("model_name must be set if stage is 'train'")
        config["accelerator"] = "gpu"
        config["devices"] = torch.cuda.device_count()
        config["strategy"] = "ddp"
        config["max_epochs"] = max_epochs
        config["gradient_clip_val"] = DEFAULT_MAX_GRAD_NORM
        config["callbacks"] = [
            # EMACallback(),
            # pl.callbacks.EarlyStopping(
            #     monitor="val/acc", patience=25, mode="max"
            # ),
            pl.callbacks.ModelCheckpoint(
                save_top_k=(-1 if save_all_checkpoints else 1),
                monitor="val/acc",
                mode="max",
                every_n_epochs=1,
            ),
            pl.callbacks.TQDMProgressBar(),
        ]
        config["logger"] = [
            pl.loggers.TensorBoardLogger(
                str(output_dir / "tb_logs"),
                name=model_name,
                default_hp_metric=False,
                version=version,
            ),
            pl.loggers.CSVLogger(
                str(output_dir / "csv_logs"),
                name=model_name,
                version=version,
            ),
        ]
    else:
        config["devices"] = 1
        config["num_nodes"] = 1
    return pl.Trainer(**config)  # type: ignore


def train(
    model_name: str,
    dataset_name: str,
    output_dir: Path | str,
    ckpt_path: Path | None = None,
    ce_weight: float = 1,
    lcc_submodules: list[str] | None = None,
    lcc_kwargs: dict | None = None,
    max_epochs: int = 50,
    batch_size: int = 256,
    train_split: str = "train",
    val_split: str = "val",
    test_split: str | None = None,
    image_key: str = "image",
    label_key: str = "label",
    logit_key: str | None = "logits",
    head_name: str | None = None,
    seed: int | None = None,
) -> dict:
    """
    Performs fine-tuning on a model, possibly with latent clustering correction.

    Args:
        model_name (str): The model name as in the [Hugging Face model
            hub](https://huggingface.co/models?pipeline_tag=image-classification).
        dataset_name (str): The dataset name as in the [Hugging Face dataset
            hub](https://huggingface.co/datasets?task_categories=task_categories:image-classification).
        output_dir (Path | str):
        ckpt_path (Path | None): If `None`, the correction will start from the
            weights available on the Hugging Face model hub.
        ce_weight (float, optional): Weight of the cross-entropy loss against
            the LCC loss. Ignored if LCC is not performed. Defaults to $1$.
        lcc_submodules (list[str] | None, optional): List of submodule names
            where to perform LCC. If empty or `None`, LCC is not performed. This
            is the only way to enable/disable LCC. Defaults to `None`.
        lcc_kwargs (dict | None, optional): Optional parameters for LCC. See
            `lcc.classifiers.BaseClassifier.__init__`.
        max_epochs (int, optional): Defaults to $50$.
        batch_size (int, optional): Defaults to $2048$.
        train_split (str, optional):
        val_split (str, optional):
        test_split (str | None, optional):
        image_key (str, optional):
        label_key (str, optional):
        logit_key (str | None, optional):
        head_name (str | None, optional): Name of the output layer of the model.
            This must be set if the number of classes in the dataset does not
            match the number components of the output layer of the model. See
            also `lcc.classifiers.BaseClassifier.__init__`.
        seed (int | None, optional): Global seed for both CPU and GPU. If not
            `None`, this is set globally, so one might consider this as a side
            effect.
    """
    if seed is not None:
        r0_info("Setting global seed to {}", seed)
        torch.manual_seed(seed)

    lcc_kwargs, do_lcc = lcc_kwargs or {}, bool(lcc_submodules)
    if do_lcc:
        r0_info("Performing latent cluster correction")
        validate_lcc_kwargs(lcc_kwargs)

    output_dir = Path(output_dir)
    _dataset_name = dataset_name.replace("/", "-")
    _model_name = model_name.replace("/", "-")
    _output_dir = output_dir / _dataset_name / _model_name
    _output_dir.mkdir(parents=True, exist_ok=True)

    classifier_cls = get_classifier_cls(model_name)

    if dataset_name.startswith("PRESET:"):
        dataset_name = dataset_name[7:]
        r0_info("Using preset dataset name: {}", dataset_name)
        dataset, _ = get_dataset(
            dataset_name,
            image_processor=model_name,
            batch_size=batch_size,
            num_workers=get_reasonable_n_jobs(),
        )
    else:
        dataset = HuggingFaceDataset(
            dataset_name=dataset_name,
            fit_split=train_split,
            val_split=val_split,
            test_split=test_split,
            label_key=label_key,
            image_processor=classifier_cls.get_image_processor(model_name),
            train_dl_kwargs={
                "batch_size": batch_size,
                "num_workers": get_reasonable_n_jobs(),
            },
            val_dl_kwargs={
                "batch_size": batch_size,
                "num_workers": get_reasonable_n_jobs(),
            },
        )
    n_classes = dataset.n_classes()

    model = classifier_cls(
        model_name=model_name,
        n_classes=n_classes,
        head_name=head_name,
        image_key=image_key,
        label_key=label_key,
        logit_key=logit_key,
        lcc_submodules=lcc_submodules if do_lcc else None,
        lcc_kwargs=lcc_kwargs if do_lcc else None,
        ce_weight=ce_weight,
    )
    if isinstance(ckpt_path, Path):
        model.model = classifier_cls.load_from_checkpoint(  # type: ignore
            ckpt_path
        ).model
        r0_info("Loaded checkpoint {}", ckpt_path)
    r0_debug("Model hyperparameters:\n{}", json.dumps(model.hparams, indent=4))

    trainer = make_trainer(
        _output_dir,
        model_name=_model_name,
        max_epochs=max_epochs,
        stage="train",
        version=str(uuid.uuid4().hex),
    )
    start = datetime.now()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        trainer.fit(model, dataset)
    fit_time = datetime.now() - start
    r0_info("Finished training in {}", fit_time)

    ckpt = Path(trainer.checkpoint_callback.best_model_path)  # type: ignore
    ckpt = ckpt.relative_to(output_dir)
    v, e, s = checkpoint_ves(ckpt)
    r0_info("Best checkpoint path: {}", ckpt)
    r0_info("version={}, best_epoch={}, n_steps={}", v, e, s)

    # TODO: fix testing loop. Right now, every rank reinstanciates a single-node
    # single-device trainer to run the model on the test dataset. So every rank
    # is testing the model independently which is stupid.

    with TemporaryDirectory(prefix="lcc-") as tmp:
        trainer = make_trainer(tmp, stage="test")
        test_results = trainer.test(model, dataset)

    document: dict = {
        "__meta__": {
            "version": 3,
            "hostname": os.uname().nodename,
            "datetime": start,
        },
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
        "model": {"name": model_name, "hparams": dict(model.hparams)},
        "training": {
            "best_checkpoint": {
                "path": str(ckpt),
                "version": v,
                "epoch": e,
                "n_steps": s,
            },
            "seed": seed,
            "time": fit_time / timedelta(seconds=1),
            "test": test_results,
        },
    }
    document["__meta__"]["hash"] = _dict_sha1(
        {k: document[k] for k in ["dataset", "model"]}
    )
    tb.save_json(document, _output_dir / f"results.{v}.json")
    return document
