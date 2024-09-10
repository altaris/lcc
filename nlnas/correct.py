"""Latent clustering correction"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import torch
import turbo_broccoli as tb
from loguru import logger as logging

from .classifiers import BaseClassifier, HuggingFaceClassifier, TimmClassifier
from .datasets import HuggingFaceDataset
from .finetune import make_trainer
from .logging import r0_info
from .training import checkpoint_ves


# pylint: disable=duplicate-code
def correct(
    model_name: str,
    ckpt_path: Path | None,
    dataset_name: str,
    output_dir: Path,
    lcc_submodules: list[str],
    clst_weight: float = 1,
    ce_weight: float = 1,
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
    Performs latent clustering correction on a pretrained model.

    Args:
        model_name (str):
        ckpt_path (Path | None): If `None`, the correction will start from the
            weights available on the Hugging Face model hub.
        dataset_name (str):
        output_dir (Path):
        lcc_submodules (list[str]):
        clst_weight (float, optional):
        ce_weight (float, optional):
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
    if (n_cuda_devices := torch.cuda.device_count()) > 1:
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
        clst_weight=clst_weight,
        lcc_submodules=lcc_submodules,
        ce_weight=ce_weight,
    )
    if isinstance(ckpt_path, Path):
        # pylint: disable=no-value-for-parameter
        model.model = classifier_cls.load_from_checkpoint(  # type: ignore
            ckpt_path
        ).model
        r0_info("Loaded checkpoint '{}'", ckpt_path)

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
        "model": {"name": model_name},
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
        "correction": {
            "hparams": dict(model.hparams),
            "max_epochs": max_epochs,
            "lcc_submodules": lcc_submodules,
            "clst_weight": clst_weight,
            "ce_weight": ce_weight,
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
