# NLNAS

![Python 3](https://img.shields.io/badge/python-3-blue?logo=python)
[![License](https://img.shields.io/badge/license-MIT-green)](https://choosealicense.com/licenses/mit/)

## Installation

Make sure [`uv`](https://docs.astral.sh/uv/) is installed. Then run

```sh
uv python install 3.10
uv sync --all-extras
```

## Usage

- Pretty-print a model structure: run `./pretty-print.sh HF_MODEL_NAME`, e.g. `./pretty-print.sh microsoft/resnet-18`
- Fine-tuning: modify and run `finetune.sh`
- Latent cluster correction: modify and run `correct.sh`
