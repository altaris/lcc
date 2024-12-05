# LCC: Latent Cluster Correction

![Python 3.10](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![CUDA 12](https://img.shields.io/badge/CUDA-12-green?logo=nvidia)
[![Documentation](https://img.shields.io/badge/docs-here-pink)](https://cedric.hothanh.fr/lcc/lcc.html)
[![License](https://img.shields.io/badge/license-MIT-white)](https://choosealicense.com/licenses/mit/)

- Neural networks take input samples and transform them into **latent
  representations**
- Semantically similar samples tend to aggregate into **latent clusters**
- This repository implements **Latent Cluster Correction**, a new technique to
  improve said latent clusters
- I don't want to write more academic blabla rn
- I'll link the article somedays

![](docs/imgs/microsoft-resnet-18_cifar10.png)

<!-- ![](docs/imgs/timm-vgg11.tv_in1k_timm-eurosat-rgb.png) -->
<!-- ![](docs/imgs/alexnet_microsoft-cats_vs_dogs.png) -->

## Installation

Make sure [`uv`](https://docs.astral.sh/uv/) is installed. Then run

```sh
uv python install 3.10
uv sync --all-extras
```

## Usage

- Fine-tuning with LCC: modify and run `lcc.sh`, or use the CLI directly:

  ```sh
  uv run python -m lcc train --help
  ```

  For example:

  ```sh
  uv run python -m lcc train \
    microsoft/resnet-18 \
    PRESET:cifar100 \
    output_dir \
    --batch-size 256 \
    --head-name classifier.1 \
    --logit-key logits \
    --lcc-submodules resnet.encoder.stages.3 \
    --lcc-warmup 1 \
    --lcc-weight 0.01 \
    --seed 123
  ```

- Pretty-print a model structure from
  [HuggingFace](https://huggingface.co/models?pipeline_tag=image-classification):
  run `./pretty-print.sh HF_MODEL_NAME`, e.g.

  ```sh
  ./pretty-print.sh microsoft/resnet-18
  ```

## API overview

- [`lcc.training`](https://cedric.hothanh.fr/lcc/lcc/training.html): Training
  stuff
  - [`lcc.training.train`](https://cedric.hothanh.fr/lcc/lcc/training.html#train):
    Pulls and trains a model from the [HuggingFace model
    hub](https://huggingface.co/models?pipeline_tag=image-classification)
    (presumably pretrained on ImageNet) on a dataset also pulled from
    [HuggingFace](https://huggingface.co/datasets?task_categories=task_categories:image-classification).
    This method takes the model and dataset name as argument, so it's pretty
    rigid.
- [`lcc.datasets`](https://cedric.hothanh.fr/lcc/lcc/datasets.html): Dataset
  stuff

  - [`lcc.datasets.HuggingFaceDataset`](https://cedric.hothanh.fr/lcc/lcc/datasets.html#HuggingFaceDataset):
    A HuggingFace image classification dataset wrapped inside a [Lightning
    Datamodule](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningDataModule.html)
    for easy use with PyTorch Lightning.
  - [`lcc.datasets.get_dataset`](https://cedric.hothanh.fr/lcc/lcc/datasets.html#get_dataset):
    Creating a `HuggingFaceDataset` required a bunch of arguments. I was tired
    of copy-pasting them around, so I made this method to create classical
    datasets more quickly. See
    [`nlnas.datasets.DATASET_PRESETS_CONFIGURATIONS`](https://github.com/altaris/lcc/blob/728df7ef3124fba5c74343a528dfb8160822f3b7/lcc/datasets/preset.py#L10C30-L10C31)
    for the list of available presets.

- [`lcc.classifiers`](https://cedric.hothanh.fr/lcc/lcc/classifiers.html):
  Classifier models and wrappers
  - [`lcc.classifiers.HuggingFaceClassifier`](https://cedric.hothanh.fr/lcc/lcc/classifiers.html#HuggingFaceClassifier):
    A HuggingFace image classification model wrapped inside a [Lightning
    Module](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html)
    for easy use with PyTorch Lightning.
  - [`lcc.classifiers.TimmClassifier`](https://cedric.hothanh.fr/lcc/lcc/classifiers.html#TimmClassifier):
    Same but for [`timm` models](https://huggingface.co/docs/timm/index), which
    despite also coming from the Huggingface hub, require some special
    considerations. See also [`timm.list_models`](https://huggingface.co/docs/timm/reference/models#timm.list_models).
- [`lcc.correction`](https://cedric.hothanh.fr/lcc/lcc/correction.html): LCC
  stuff. You probably don't need to touch that directly since LCC is done
  automatically for classifier classes found in
  [`lcc.classifiers`](https://cedric.hothanh.fr/lcc/lcc/classifiers.html).
- [`lcc.plotting`](https://cedric.hothanh.fr/lcc/lcc/plotting.html): Cool
  plotting stuff.
  - [`lcc.plotting.class_scatter`](https://cedric.hothanh.fr/lcc/lcc/plotting.html#class_scatter):
    2D scatter plot where samples are colored by class. Also support "outliers",
    which are samples with negative label.
