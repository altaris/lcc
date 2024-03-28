"""
Simple script to validate a model (pulled from [Hugging
Face](https://huggingface.co/models?pipeline_tag=image-classification&dataset=dataset:imagenet-1k))
against [`imagenet-1k`](https://huggingface.co/datasets/imagenet-1k).
"""

import pytorch_lightning as pl
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

from nlnas import HuggingFaceDataset, WrappedClassifier

HF_CACHE_DIR = "/export/users/hothanh/huggingface/datasets"
MODEL_NAME = "microsoft/resnet-50"

if __name__ == "__main__":
    # For performance, depending on your GPU
    torch.set_float32_matmul_precision("medium")
    dataset = HuggingFaceDataset(
        "imagenet-1k",
        # Name of the split containing training/validation data. See e.g. the
        # dataset viewer https://huggingface.co/datasets/imagenet-1k . It is
        # also possible to use more advanced syntax, see
        # https://huggingface.co/docs/datasets/en/loading#slice-splits
        fit_split="train",
        val_split="validation",
        image_processor=AutoImageProcessor.from_pretrained(MODEL_NAME),
        cache_dir=HF_CACHE_DIR,
    )
    model = WrappedClassifier(
        AutoModelForImageClassification.from_pretrained(MODEL_NAME),
        n_classes=1000,
        # Name of the columns in the dataset (see the dataset viewer again)
        image_key="image",
        label_key="label",
        # The MODEL does not return logits directly but some sort of dataclass.
        # This key allows to retrieve the logits from the output.
        logit_key="logits",
    )
    trainer = pl.Trainer()
    results = trainer.validate(model, dataset)
