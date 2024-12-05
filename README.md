# LCC: Latent Cluster Correction

![Python 3.10](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![CUDA 12](https://img.shields.io/badge/CUDA-12-green?logo=python)
[![License](https://img.shields.io/badge/license-MIT-white)](https://choosealicense.com/licenses/mit/)

![](docs/imgs/microsoft-resnet-18_cifar10.png)

<!-- ![a](docs/imgs/timm-vgg11.tv_in1k_timm-eurosat-rgb.png) -->
<!-- ![a](docs/imgs/alexnet_microsoft-cats_vs_dogs.png) -->

## Installation

Make sure [`uv`](https://docs.astral.sh/uv/) is installed. Then run

```sh
uv python install 3.10
uv sync --all-extras
```

## Usage

- Fine-tuning with LCC: modify and run `lcc.sh`, or use the CLI directly:

  ```sh
  uv run python -m nlnas train --help
  ```

  For example:

  ```sh
  uv run python -m nlnas train \
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
