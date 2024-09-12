"""Classical fine-tuning of HuggingFace models on HuggingFace datasets"""

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pytorch_lightning as pl
import turbo_broccoli as tb

from .classifiers import BaseClassifier, HuggingFaceClassifier, TimmClassifier
from .datasets import HuggingFaceDataset
from .logging import r0_info
from .training import NoCheckpointFound, best_checkpoint_path, checkpoint_ves

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
):
    """
    Loads and fine-tunes a pretrained HuggingFace model on a HuggingFace
    datasets.

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
    """
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
        train_dl_kwargs={
            "batch_size": batch_size,
            "num_workers": 16,
            "persistent_workers": True,
            "pin_memory": False,
        },
    )
    n_classes = dataset.n_classes()

    try:
        ckpt, _ = best_checkpoint_path(_output_dir)
        # pylint: disable=no-value-for-parameter
        model = classifier_cls.load_from_checkpoint(ckpt)  # type: ignore
        v, e, s = checkpoint_ves(ckpt)
        r0_info("Found best checkpoint: '{}'", ckpt)
        r0_info("version={}, best_epoch={}, n_steps={}", v, e, s)
    except NoCheckpointFound:
        r0_info(
            "No checkpoint found in {}, starting fine-tuning from scratch",
            _output_dir,
        )
        model = classifier_cls(
            model_name=model_name,
            n_classes=n_classes,
            head_name=head_name,
            image_key=image_key,
            label_key=label_key,
            logit_key=logit_key,
            optimizer="adam",
            optimizer_kwargs={"lr": 5e-5},
            # scheduler="linearlr",
        )
        trainer = make_trainer(_model_name, _output_dir, max_epochs)
        start = datetime.now()
        trainer.fit(model, dataset)
        fit_time = datetime.now() - start
        r0_info("Finished fine-tuning in {}", fit_time)
        ckpt = Path(trainer.checkpoint_callback.best_model_path)  # type: ignore
        ckpt = ckpt.relative_to(output_dir)
        v, e, s = checkpoint_ves(ckpt)
        r0_info("Best checkpoint path: '{}'", ckpt)
        r0_info("version={}, best_epoch={}, n_steps={}", v, e, s)

    with TemporaryDirectory() as tmp:
        trainer = pl.Trainer(
            callbacks=pl.callbacks.TQDMProgressBar(),
            default_root_dir=tmp,
            devices=1,
        )
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
        "fine_tuning": {
            "hparams": dict(model.hparams),
            "max_epochs": max_epochs,
            "best_checkpoint": {
                "path": str(ckpt),
                "version": v,
                "best_epoch": e,
                "n_steps": s,
            },
            # "time": fit_time / timedelta(seconds=1),
            "test": test_results,
        },
    }
    tb.save_json(document, _output_dir / "results.json")


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
                monitor="val/loss", patience=10, mode="min"
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
