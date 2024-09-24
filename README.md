# NLNAS

![Python 3](https://img.shields.io/badge/python-3-blue?logo=python)
[![License](https://img.shields.io/badge/license-MIT-green)](https://choosealicense.com/licenses/mit/)
[![Code style](https://img.shields.io/badge/style-black-black)](https://pypi.org/project/black)

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

## Contributing

### Dependencies

- `python3.10`,
- [`uv`](https://docs.astral.sh/uv/),
- `make` (optional).

### Documentation

Simply run

```sh
make docs
```

This will generate the HTML doc of the project, and the index file should be at
`docs/index.html`. To have it directly in your browser, run

```sh
make docs-browser
```

### Code quality

Don't forget to run

```sh
make
```

to format and check the code using [`ruff`](https://docs.astral.sh/ruff/) and
typecheck it using [mypy](http://mypy-lang.org/).
