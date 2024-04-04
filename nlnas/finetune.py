"""
Fine-tuning of HuggingFace models on HuggingFace datasets.
"""

from pathlib import Path

import pytorch_lightning as pl
from transformers import AutoImageProcessor, AutoModelForImageClassification

from .classifiers import WrappedClassifier
from .datasets import HuggingFaceDataset

DEFAULT_MAX_GRAD_NORM = 1.0


def finetune(
    model_name: str,
    dataset_name: str,
    n_classes: int,
    output_dir: Path,
    epochs: int = 10,
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
    Loads and fine-tunes a pretrained HuggingFace model on a HuggingFace
    datasets.

    Args:
        model_name (str): See the [HuggingFace model
            index](https://huggingface.co/models?pipeline_tag=image-classification)
        dataset_name (str): See the [HuggingFace dataset
            index](https://huggingface.co/datasets?task_categories=task_categories:image-classification)
        n_classes (int): Number of classes in the dataset. Sadly this is not
            computed automatically ^^"
        output_dir (Path):
        epochs (int, optional):
        batch_size (int, optional):
        train_split (str, optional):
        val_split (str, optional):
        test_split (str, optional):
        image_key (str, optional):
        label_key (str, optional):
        logit_key (str, optional):
        head_name (str | None, optional):
    """
    _dataset_name = dataset_name.replace("/", "-")
    _model_name = model_name.replace("/", "-")
    output_dir = output_dir / _dataset_name / _model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    tb_logger = pl.loggers.TensorBoardLogger(
        str(output_dir / "tb_logs"), name=_model_name
    )
    csv_logger = pl.loggers.CSVLogger(
        str(output_dir / "csv_logs"), name=_model_name
    )
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                save_top_k=1, monitor="val/acc", mode="max", every_n_epochs=1
            ),
            pl.callbacks.TQDMProgressBar(),
        ],
        default_root_dir=str(output_dir),
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

    model = WrappedClassifier(
        model=AutoModelForImageClassification.from_pretrained(model_name),
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
    )

    trainer.fit(model, dataset)
    trainer.test(model, dataset)
