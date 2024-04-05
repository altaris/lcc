"""
Simple script to validate a model (pulled from
[HuggingFace](https://huggingface.co/models?pipeline_tag=image-classification))
against some dataset, also from
[HuggingFace](https://huggingface.co/datasets?task_categories=task_categories:image-classification).
"""

import pytorch_lightning as pl
import torch
from transformers import AutoModelForImageClassification

from nlnas import HuggingFaceDataset, WrappedClassifier

HF_MODEL_NAME = "microsoft/resnet-50"

HF_DATASET_NAME = "imagenet-1k"
N_CLASSES = 1000  # Sadly this isn't computed automatically :(

# Name of the split containing training/validation data. See e.g. the dataset
# viewer https://huggingface.co/datasets/imagenet-1k . It is also possible to
# use more advanced syntax, see
# https://huggingface.co/docs/datasets/en/loading#slice-splits
FIT_SPLIT = "train"
VAL_SPLIT = "validation"

# Name of the columns in the dataset (see the dataset viewer again)
IMAGE_KEY = "image"
LABEL_KEY = "label"

# The model does not return logits directly but some sort of dataclass. This
# key allows to retrieve the logits from the output. If the output is a
# ImageClassifierOutput
# (https://huggingface.co/docs/transformers/v4.39.3/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput)
# or a ImageClassifierOutputWithNoAttention
# (https://huggingface.co/docs/transformers/v4.39.3/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput),
# then this key should be "logits" (the default).
LOGIT_KEY = "logits"

if __name__ == "__main__":
    # For performance, depending on your GPU
    torch.set_float32_matmul_precision("medium")
    dataset = HuggingFaceDataset(
        HF_DATASET_NAME,
        fit_split=FIT_SPLIT,
        val_split=VAL_SPLIT,
        image_processor=HF_MODEL_NAME,
    )
    model = WrappedClassifier(
        AutoModelForImageClassification.from_pretrained(HF_MODEL_NAME),
        n_classes=N_CLASSES,
        image_key=IMAGE_KEY,
        label_key=LABEL_KEY,
        logit_key=LOGIT_KEY,
    )
    trainer = pl.Trainer()
    results = trainer.validate(model, dataset)
