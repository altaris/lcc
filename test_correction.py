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
import turbo_broccoli as tb
from loguru import logger as logging
from torch import Tensor
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification

from nlnas import (
    HuggingFaceDataset,
    WrappedClassifier,
    max_connected_confusion_choice,
)

HF_DATASET_NAME = "imagenet-1k"  # Name in Hugging Face's dataset index
HF_MODEL_NAME = "microsoft/resnet-50"  # Name in Hugging Face's model index

N_CLASSES = 1000  # Number of classes in the dataset
TRAIN_SPLIT = "train"  # See HF dataset page for split name
VAL_SPLIT = "validation"  # See HF dataset page for split name
TEST_SPLIT = "validation"  # See HF dataset page for split name
IMAGE_KEY = "image"  # See HF dataset page for name of dataset column
LABEL_KEY = "label"  # See HF dataset page for name of dataset column
LOGIT_KEY = "logits"  # See HF dataset page for name of dataset column

# Filesystem-friendly names
DATASET_NAME = HF_DATASET_NAME.replace("/", "-")
MODEL_NAME = HF_MODEL_NAME.replace("/", "-")

OUTPUT_DIR = Path("out.local") / DATASET_NAME / MODEL_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CORRECTION_SUBMODULES = [  # See also `nlnas.utils.pretty_print_submodules`
    # "model.resnet.encoder.stages.0",
    # "model.resnet.encoder.stages.1",
    # "model.resnet.encoder.stages.2",
    "model.resnet.encoder.stages.3",
]
CORRECTION_WEIGHT = 1e-5
N_CORRECTIONS_EPOCHS = 100
N_CLASSES_PER_CORRECTION_EPOCH = 5

# HF_CACHE_DIR = "/export/users/hothanh/huggingface/datasets"


def get_val_y_true() -> Tensor:
    """
    Gets the label vector of the validation dataset. This method is internally
    guarded and the artefact is `OUTPUT_DIR/../y_true.val.st`.
    """
    h = tb.GuardedBlockHandler(OUTPUT_DIR.parent / "y_true.val.st")
    for _ in h:
        dataset = HuggingFaceDataset(
            HF_DATASET_NAME,
            fit_split=TRAIN_SPLIT,
            val_split=VAL_SPLIT,
            image_processor=AutoImageProcessor.from_pretrained(HF_MODEL_NAME),
        )
        dataset.prepare_data()
        dataset.setup("fit")
        y_true = torch.concat(
            [
                batch[LABEL_KEY]
                for batch in tqdm(
                    dataset.val_dataloader(),
                    desc="Constructing label vector",
                    leave=False,
                )
            ]
        )
        h.result = {"y_true": y_true}
    # TB loads safetensor files as numpy arrays...
    return Tensor(h.result["y_true"])


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
        image_key=IMAGE_KEY,
        label_key=LABEL_KEY,
        logit_key=LOGIT_KEY,
        cor_submodules=CORRECTION_SUBMODULES,
        cor_weight=CORRECTION_WEIGHT,
    )
    logging.debug("Loaded model: {}", HF_MODEL_NAME)

    y_true = get_val_y_true()
    trainer = make_trainer()
    for epoch in range(1, N_CORRECTIONS_EPOCHS + 1):
        logging.info(
            "Starting correction epoch {}/{}", epoch, N_CORRECTIONS_EPOCHS
        )

        # COMPUTE PREDUCTIONS ON VAL DATASET
        h = tb.GuardedBlockHandler(OUTPUT_DIR / f"y_pred.val.{epoch}.st")
        for _ in h:
            dataset = HuggingFaceDataset(
                HF_DATASET_NAME,
                fit_split=TRAIN_SPLIT,
                val_split=VAL_SPLIT,
                predict_split=VAL_SPLIT,
                image_processor=AutoImageProcessor.from_pretrained(
                    HF_MODEL_NAME
                ),
                # cache_dir=HF_CACHE_DIR,
            )
            results = trainer.predict(model, dataset)
            y_pred = torch.concat(results)  # type: ignore
            h.result = {"y_pred": y_pred}
        y_pred = Tensor(h.result["y_pred"])

        # SELECT CORRECTION CLASSES
        classes, confusion = max_connected_confusion_choice(
            y_pred, y_true, 1000, N_CLASSES_PER_CORRECTION_EPOCH
        )
        logging.debug("Selected classes: {}", classes)
        logging.debug("Total confusion: {}", confusion)
        with (OUTPUT_DIR / f"classes.{epoch}.json").open(
            "w", encoding="utf-8"
        ) as fp:
            json.dump(
                {"classes": classes, "confusion": confusion},
                fp,
            )

        # LOAD FILTERED DATASET
        dataset = HuggingFaceDataset(
            HF_DATASET_NAME,
            fit_split=TRAIN_SPLIT,
            val_split=VAL_SPLIT,
            test_split=TEST_SPLIT,
            image_processor=AutoImageProcessor.from_pretrained(HF_MODEL_NAME),
            # cache_dir=HF_CACHE_DIR,
            classes=classes,
        )
        logging.debug("Loaded dataset '{}'", HF_DATASET_NAME)

        # FINE TUNE
        trainer.fit_loop.max_epochs = epoch
        trainer.fit(model, dataset)

        # TEST (logs to tensorboard)
        trainer.test(model, dataset)

        # del dataset
        # torch.cuda.empty_cache()

    logging.info("Finished fine-tuning in {}", datetime.now() - start)
