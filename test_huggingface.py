"""
Simple script to validate a model (pulled from [Hugging
Face](https://huggingface.co/models?pipeline_tag=image-classification&dataset=dataset:imagenet-1k))
on [`imagenet-1k`](https://huggingface.co/datasets/imagenet-1k).
"""

import pytorch_lightning as pl
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

from nlnas import HuggingFaceDataset, WrappedClassifier

HF_CACHE_DIR = "/export/users/hothanh/huggingface/datasets"
MODEL = "microsoft/resnet-50"

if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    dataset = HuggingFaceDataset(
        "imagenet-1k",
        fit_split="train",
        val_split="validation",
        image_processor=AutoImageProcessor.from_pretrained(MODEL),
        # cache_dir=HF_CACHE_DIR,
    )
    model = WrappedClassifier(
        AutoModelForImageClassification.from_pretrained(MODEL),
        n_classes=1000,
        image_key="image",
        label_key="label",
        logit_key="logits",
    )
    trainer = pl.Trainer()
    results = trainer.validate(model, dataset)
