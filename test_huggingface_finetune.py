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

from nlnas import HuggingFaceDataset, WrappedClassifier, train_model_guarded

HF_CACHE_DIR = "/export/users/hothanh/huggingface/datasets"
MODEL_NAME = "microsoft/resnet-50"
OUTPUT_DIR = Path("out")

N_CORRECTIONS_EPOCHS = 100
N_CLASSES_PER_CORRECTION_EPOCH = 10


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


if __name__ == "__main__":
    pl.seed_everything(0)
    torch.set_float32_matmul_precision("medium")

    start = datetime.now()

    submodules = [
        # "model.resnet.encoder.stages.0",
        # "model.resnet.encoder.stages.1",
        # "model.resnet.encoder.stages.2",
        "model.resnet.encoder.stages.3",
    ]
    model = WrappedClassifier(
        AutoModelForImageClassification.from_pretrained(MODEL_NAME),
        n_classes=1000,
        image_key="image",
        label_key="label",
        logit_key="logits",
        cor_submodules=submodules,
        cor_weight=1e-3,
    )
    name = MODEL_NAME.replace("/", "_")
    logging.info("Loaded model: {}", MODEL_NAME)

    for epoch in range(1, N_CORRECTIONS_EPOCHS + 1):
        logging.info(
            "Starting correction epoch {}/{}", epoch, N_CORRECTIONS_EPOCHS
        )

        output_dir = OUTPUT_DIR / name / str(epoch)
        output_dir.mkdir(parents=True, exist_ok=True)

        classes = choice(
            torch.arange(1000), N_CLASSES_PER_CORRECTION_EPOCH
        ).tolist()
        with (output_dir / "classes.json").open("w", encoding="utf-8") as fp:
            json.dump(classes, fp)
        dataset = HuggingFaceDataset(
            "imagenet-1k",
            fit_split="train",
            val_split="validation",
            image_processor=AutoImageProcessor.from_pretrained(MODEL_NAME),
            cache_dir=HF_CACHE_DIR,
            classes=classes,
        )
        logging.info("Loaded imagenet-1k")
        logging.info(
            "Selected {} classes: {}", N_CLASSES_PER_CORRECTION_EPOCH, classes
        )
        logging.info("Filtered dataset size: {}", len(dataset))

        train_model_guarded(
            model,
            dataset,
            OUTPUT_DIR / name / str(epoch),
            name,
            max_epochs=1,
        )

    logging.info("Finished fine-tuning in {}", datetime.now() - start)
