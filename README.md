# NLNAS

![Python 3](https://img.shields.io/badge/python-3-blue?logo=python)
[![License](https://img.shields.io/badge/license-MIT-green)](https://choosealicense.com/licenses/mit/)
[![Code style](https://img.shields.io/badge/style-black-black)](https://pypi.org/project/black)

## Usage

### Fine-tuning

```sh
python3.10 -m nlnas finetune \
    microsoft/resnet-18 cifar100 out.local/ft \
    --train-split 'train[:80%]' \
    --val-split 'train[80%:]' \
    --test-split test \
    --image-key img \
    --label-key fine_label \
    --head-name classifier.1
```

### Latent clustering correction

(note that for now, CUDA cannot be used for LCC)

```sh
CUDA_VISIBLE_DEVICES=
FILE=out/ft/cifar100/microsoft-resnet-18/results.0.json
python3.10 -m nlnas correct \
    $(jq -r .model.name < $FILE) \
    $(jq -r .dataset.name < $FILE) \
    resnet.encoder.stages.3,classifier.1 \
    0.001 \
    out/lcc \
    --ckpt-path out/ft/$(jq -r .fine_tuning.best_checkpoint.path < $FILE) \
    --train-split $(jq -r .dataset.train_split < $FILE) \
    --val-split $(jq -r .dataset.val_split < $FILE) \
    --test-split $(jq -r .dataset.test_split < $FILE) \
    --image-key $(jq -r .dataset.image_key < $FILE) \
    --label-key $(jq -r .dataset.label_key < $FILE) \
    --head-name $(jq -r .fine_tuning.hparams.head_name < $FILE)
```

```sh
CUDA_VISIBLE_DEVICES=
FILE=out/ft/cifar100/google-mobilenet_v2_1.0_224/results.0.json
python3.10 -m nlnas correct \
    $(jq -r .model.name < $FILE) \
    $(jq -r .dataset.name < $FILE) \
    model.mobilenet_v2.layer.15,model.mobilenet_v2.conv_1x1,model.classifier \
    0.01 \
    out/lcc \
    --ckpt-path out/ft/$(jq -r .fine_tuning.best_checkpoint.path < $FILE) \
    --train-split $(jq -r .dataset.train_split < $FILE) \
    --val-split $(jq -r .dataset.val_split < $FILE) \
    --test-split $(jq -r .dataset.test_split < $FILE) \
    --image-key $(jq -r .dataset.image_key < $FILE) \
    --label-key $(jq -r .dataset.label_key < $FILE) \
    --head-name $(jq -r .fine_tuning.hparams.head_name < $FILE)
```

## Contributing

### Dependencies

- `python3.10` or newer;
- `requirements.txt` for runtime dependencies;
- `requirements.dev.txt` for development dependencies.
- `make` (optional);

Simply run

```sh
virtualenv venv -p python3.10
. ./venv/bin/activate
python3.10 -m pip install --upgrade pip
python3.10 -m pip install -r requirements.txt
python3.10 -m pip install -r requirements.dev.txt
python3.10 -m pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com -r requirements.cuda12.txt
```

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

to format the code following [black](https://pypi.org/project/black/),
typecheck it using [mypy](http://mypy-lang.org/), and check it against coding
standards using [pylint](https://pylint.org/).
