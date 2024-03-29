"""
Simple script to finetune a model (pulled from [Hugging
Face](https://huggingface.co/models?pipeline_tag=image-classification&dataset=dataset:imagenet-1k))
against [`imagenet-1k`](https://huggingface.co/datasets/imagenet-1k) using
Louvain clustering correction.

See `test_huggingface_val.py` for some helpful comments.
"""

import json
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch
from loguru import logger as logging
from transformers import AutoImageProcessor, AutoModelForImageClassification

from nlnas import HuggingFaceDataset, WrappedClassifier

HF_MODEL_NAME = "microsoft/resnet-50"  # Name in Hugging Face's model index
MODEL_NAME = HF_MODEL_NAME.replace("/", "-")
OUTPUT_DIR = Path("out.local") / MODEL_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CORRECTION_SUBMODULES = [  # See also `nlnas.utils.pretty_print_submodules`
    # "model.resnet.encoder.stages.0",
    # "model.resnet.encoder.stages.1",
    # "model.resnet.encoder.stages.2",
    "model.resnet.encoder.stages.3",
]
CORRECTION_WEIGHT = 1e-3
N_CORRECTIONS_EPOCHS = 100
N_CLASSES_PER_CORRECTION_EPOCH = 10

HF_CACHE_DIR = "/export/users/hothanh/huggingface/datasets"


def choice(
    a: torch.Tensor,
    n: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Analogous to
    [`numpy.random.choice`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html)
    except the selection is without replacement the selection distribution is
    uniform.

    Args:
        a (torch.Tensor): Tensor to sample from.
        n (int | None, optional): Number of samples to draw. If `None`, returns
            a permutation of `a`
        generator (torch.Generator | None, optional):
    """
    idx = torch.randperm(len(a), generator=generator)
    if n is not None:
        idx = idx[:n]
    return a[idx]


def make_trainer() -> pl.Trainer:
    """Self-explanatory."""
    tb_logger = pl.loggers.TensorBoardLogger(
        str(OUTPUT_DIR / "tb_logs"), name=MODEL_NAME
    )
    csv_logger = pl.loggers.CSVLogger(
        str(OUTPUT_DIR / "csv_logs"), name=MODEL_NAME
    )
    return pl.Trainer(
        max_epochs=1,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                save_top_k=-1, monitor="val/acc", mode="max", every_n_epochs=1
            ),
            pl.callbacks.TQDMProgressBar(),
        ],
        default_root_dir=str(OUTPUT_DIR),
        logger=[tb_logger, csv_logger],
        log_every_n_steps=1,
    )


if __name__ == "__main__":
    pl.seed_everything(0)
    torch.set_float32_matmul_precision("medium")

    start = datetime.now()

    model = WrappedClassifier(
        AutoModelForImageClassification.from_pretrained(HF_MODEL_NAME),
        n_classes=1000,
        image_key="image",
        label_key="label",
        logit_key="logits",
        cor_submodules=CORRECTION_SUBMODULES,
        cor_weight=CORRECTION_WEIGHT,
    )
    logging.debug("Loaded model: {}", HF_MODEL_NAME)

    trainer = make_trainer()
    for epoch in range(1, N_CORRECTIONS_EPOCHS + 1):
        logging.info(
            "Starting correction epoch {}/{}", epoch, N_CORRECTIONS_EPOCHS
        )

        classes = choice(
            torch.arange(1000), N_CLASSES_PER_CORRECTION_EPOCH
        ).tolist()
        with (OUTPUT_DIR / f"classes.{epoch}.json").open(
            "w", encoding="utf-8"
        ) as fp:
            json.dump(classes, fp)
        dataset = HuggingFaceDataset(
            "imagenet-1k",
            fit_split="train",
            val_split="validation",
            test_split="validation",
            image_processor=AutoImageProcessor.from_pretrained(HF_MODEL_NAME),
            # cache_dir=HF_CACHE_DIR,
            classes=classes,
        )
        logging.debug("Loaded imagenet-1k")
        logging.debug(
            "Selected {} classes: {}", N_CLASSES_PER_CORRECTION_EPOCH, classes
        )

        trainer.fit_loop.max_epochs = epoch
        trainer.fit(model, dataset)
        trainer.test(model, dataset)

        # del dataset
        torch.cuda.empty_cache()

    logging.info("Finished fine-tuning in {}", datetime.now() - start)
