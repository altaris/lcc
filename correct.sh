#!/bin/sh

# shellcheck disable=SC2046

# https://www.patorjk.com/software/taag/
# Font Name: ANSI Regular
echo
echo '██       ██████  ██████'
echo '██      ██      ██'
echo '██      ██      ██'
echo '██      ██      ██'
echo '███████  ██████  ██████'

# FILE=out/ft/cifar100/timm-mobilenetv3_small_050.lamb_in1k/results.json
FILE=out/ft/cifar100/timm-tinynet_e.in1k/results.json
LCC_SUBMODULES=blocks.6

LCC_WEIGHT=5e-3
CE_WEIGHT=1

export CUDA_VISIBLE_DEVICES=0

echo
echo "=================================================="
echo "FILE:           $FILE"
echo "LCC_SUBMODULES: $LCC_SUBMODULES"
echo "LCC_WEIGHT:     $LCC_WEIGHT"
echo "CE_WEIGHT:      $CE_WEIGHT"
echo "=================================================="
echo

uv run python -m nlnas correct \
    $(jq -r .model.name < $FILE) \
    $(jq -r .dataset.name < $FILE) \
    $LCC_SUBMODULES \
    out/lcc \
    --lcc-weight $LCC_WEIGHT --ce-weight $CE_WEIGHT \
    --ckpt-path out/ft/$(jq -r .fine_tuning.best_checkpoint.path < $FILE) \
    --train-split $(jq -r .dataset.train_split < $FILE) \
    --val-split $(jq -r .dataset.val_split < $FILE) \
    --test-split $(jq -r .dataset.test_split < $FILE) \
    --image-key $(jq -r .dataset.image_key < $FILE) \
    --label-key $(jq -r .dataset.label_key < $FILE) \
    --head-name $(jq -r .fine_tuning.hparams.head_name < $FILE)
