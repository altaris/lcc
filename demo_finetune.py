"""
A demo script to fine-tune a Hugging Face model pretrained on
[ImageNet-1k](https://huggingface.co/datasets/imagenet-1k) on a different
dataset using `nlnas` APIs.

You can also use the CLI, see the README.
"""

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from transformers import AutoModelForImageClassification

from nlnas import HuggingFaceDataset, WrappedClassifier

# ↓ Name in Hugging Face's model index
HF_MODEL_NAME = "facebook/convnext-small-224"
LOGIT_KEY = "logits"  # Key to retrieve the logits fromo the model's outputs
HEAD_NAME = "classifier"  # Name of the final FC layer (to be replaced)

HF_DATASET_NAME = "cifar100"  # Name in Hugging Face's dataset index
N_CLASSES = 100  # Number of classes in the dataset
TRAIN_SPLIT = "train[:80%]"  # See HF dataset page for split name
VAL_SPLIT = "train[80%:]"  # See HF dataset page for split name
TEST_SPLIT = "test"  # See HF dataset page for split name
IMAGE_KEY = "img"  # See HF dataset page for name of dataset column
LABEL_KEY = "fine_label"  # See HF dataset page for name of dataset column
DATALOADER_KWARGS = {
    "batch_size": 64,
    "num_workers": 8,
    "persistent_workers": True,
    "pin_memory": False,
}

N_EPOCHS = 10

# Taking defaults from
# https://huggingface.co/docs/transformers/v4.39.3/en/main_classes/trainer#transformers.TrainingArguments
OPTIMIZER = "adam"
OPTIMIZER_KWARGS = {
    "lr": 5e-5,
    "weight_decay": 0,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
}
SCHEDULER = "linearlr"
SCHEDULER_KWARGS: dict[str, Any] = {}
MAX_GRAD_NORM = 1.0

# Filesystem-friendly names
DATASET_NAME = HF_DATASET_NAME.replace("/", "-")
MODEL_NAME = HF_MODEL_NAME.replace("/", "-")
OUTPUT_DIR = Path("out.local") / "finetune" / DATASET_NAME / MODEL_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def make_trainer() -> pl.Trainer:
    """Self-explanatory."""
    tb_logger = pl.loggers.TensorBoardLogger(
        str(OUTPUT_DIR / "tb_logs"), name=MODEL_NAME
    )
    csv_logger = pl.loggers.CSVLogger(
        str(OUTPUT_DIR / "csv_logs"), name=MODEL_NAME
    )
    return pl.Trainer(
        max_epochs=N_EPOCHS,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                save_top_k=1, monitor="val/loss", mode="min", every_n_epochs=1
            ),
            pl.callbacks.TQDMProgressBar(),
        ],
        default_root_dir=str(OUTPUT_DIR),
        logger=[tb_logger, csv_logger],
        log_every_n_steps=1,
        gradient_clip_val=MAX_GRAD_NORM,
    )


if __name__ == "__main__":
    # For performance, depending on your GPU
    torch.set_float32_matmul_precision("medium")
    dataset = HuggingFaceDataset(
        dataset_name=HF_DATASET_NAME,
        fit_split=TRAIN_SPLIT,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT,
        image_processor=HF_MODEL_NAME,
        dataloader_kwargs=DATALOADER_KWARGS,
    )
    model = WrappedClassifier(
        model=AutoModelForImageClassification.from_pretrained(HF_MODEL_NAME),
        n_classes=N_CLASSES,
        head_name=HEAD_NAME,
        image_key=IMAGE_KEY,
        label_key=LABEL_KEY,
        logit_key=LOGIT_KEY,
        optimizer=OPTIMIZER,
        optimizer_kwargs=OPTIMIZER_KWARGS,
        scheduler=SCHEDULER,
        scheduler_kwargs=SCHEDULER_KWARGS,
    )
    trainer = make_trainer()
    trainer.fit(model, dataset)
    trainer.test(model, dataset)
