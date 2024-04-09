"""
Fine-tuning of HuggingFace models on HuggingFace datasets.
"""

from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import turbo_broccoli as tb
from transformers import AutoImageProcessor, AutoModelForImageClassification

from .classifiers import WrappedClassifier
from .datasets import HuggingFaceDataset
from .logging import r0_info
from .training import checkpoint_ves

DEFAULT_MAX_GRAD_NORM = 1.0


def finetune(
    model_name: str,
    dataset_name: str,
    output_dir: Path,
    max_epochs: int = 100,
    batch_size: int = 64,
    train_split: str = "train",
    val_split: str = "val",
    test_split: str = "test",
    image_key: str = "image",
    label_key: str = "label",
    logit_key: str = "logits",
    head_name: str | None = None,
    correction_weight: float = 0.0,
    correction_submodules: list[str] | None = None,
):
    """
    Loads and fine-tunes a pretrained HuggingFace model on a HuggingFace
    datasets. Set `correction_weight` and `correction_submodules` to perform
    latent clustering correction at the same time (not recommended).

    Args:
        model_name (str): See the [HuggingFace model
            index](https://huggingface.co/models?pipeline_tag=image-classification)
        dataset_name (str): See the [HuggingFace dataset
            index](https://huggingface.co/datasets?task_categories=task_categories:image-classification)
        output_dir (Path):
        max_epochs (int, optional):
        batch_size (int, optional):
        train_split (str, optional):
        val_split (str, optional):
        test_split (str, optional):
        image_key (str, optional):
        label_key (str, optional):
        logit_key (str, optional):
        head_name (str | None, optional):
        correction_weight (float):
        correction_submodules (str | list[str]):
    """
    _dataset_name = dataset_name.replace("/", "-")
    _model_name = model_name.replace("/", "-")
    _output_dir = output_dir / _dataset_name / _model_name
    _output_dir.mkdir(parents=True, exist_ok=True)

    tb_logger = pl.loggers.TensorBoardLogger(
        str(_output_dir / "tb_logs"), name=_model_name
    )
    csv_logger = pl.loggers.CSVLogger(
        str(_output_dir / "csv_logs"), name=_model_name
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="val/loss", patience=10, mode="min"
            ),
            pl.callbacks.ModelCheckpoint(
                save_top_k=1, monitor="val/loss", mode="min", every_n_epochs=1
            ),
            pl.callbacks.TQDMProgressBar(),
        ],
        default_root_dir=str(_output_dir),
        logger=[tb_logger, csv_logger],
        log_every_n_steps=1,
        gradient_clip_val=DEFAULT_MAX_GRAD_NORM,
    )

    dataset = HuggingFaceDataset(
        dataset_name=dataset_name,
        fit_split=train_split,
        val_split=val_split,
        test_split=test_split,
        image_processor=AutoImageProcessor.from_pretrained(model_name),
        dataloader_kwargs={
            "batch_size": batch_size,
            "num_workers": 8,
            "persistent_workers": True,
            "pin_memory": False,
        },
    )
    n_classes = dataset.n_classes()

    model = HuggingFaceClassifier(
        model_name=model_name,
        n_classes=n_classes,
        head_name=head_name,
        image_key=image_key,
        label_key=label_key,
        logit_key=logit_key,
        optimizer="adam",
        optimizer_kwargs={
            "lr": 5e-5,
            "weight_decay": 0,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
        },
        scheduler="linearlr",
        # scheduler_kwargs={},
        cor_weight=correction_weight,
        cor_submodules=correction_submodules,
    )

    start = datetime.now()
    trainer.fit(model, dataset)
    r0_info("Finished fine-tuning in {}", datetime.now() - start)

    ckpt = Path(trainer.checkpoint_callback.best_model_path)  # type: ignore
    ckpt = ckpt.relative_to(output_dir)
    v, e, s = checkpoint_ves(ckpt)
    r0_info("Best checkpoint path: '{}'", ckpt)
    r0_info("version={}, best_epoch={}, n_steps={}", v, e, s)

    results = trainer.test(model, dataset)

    data = {
        "model": {
            "name": model_name,
            "logit_key": logit_key,
            "head_name": head_name,
        },
        "dataset": {
            "name": dataset_name,
            "n_classes": n_classes,
            "train_split": train_split,
            "val_split": val_split,
            "test_split": test_split,
            "image_key": image_key,
            "label_key": label_key,
        },
        "epochs": max_epochs,
        "batch_size": batch_size,
        "best_checkpoint": {
            "path": str(ckpt),
            "version": v,
            "best_epoch": e,
            "n_steps": s,
        },
        "test": results,
    }
    tb.save_json(data, _output_dir / "results.json")
