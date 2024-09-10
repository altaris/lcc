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

FILE=out/ft/cifar100/timm-mobilenetv3_small_050.lamb_in1k/results.json
LCC_SUBMODULES=classifier

CLST_WEIGHT=1
CE_WEIGHT=1e-3

echo
echo "=================================================="
echo "FILE:           $FILE"
echo "LCC_SUBMODULES: $LCC_SUBMODULES"
echo "CLST_WEIGHT:    $CLST_WEIGHT"
echo "CE_WEIGHT:      $CE_WEIGHT"
echo "=================================================="
echo

export CUDA_VISIBLE_DEVICES=

python -m nlnas correct \
    $(jq -r .model.name < $FILE) \
    $(jq -r .dataset.name < $FILE) \
    $LCC_SUBMODULES \
    out/lcc \
    --clst-weight $CLST_WEIGHT --ce-weight $CE_WEIGHT \
    --ckpt-path out/ft/$(jq -r .fine_tuning.best_checkpoint.path < $FILE) \
    --train-split $(jq -r .dataset.train_split < $FILE) \
    --val-split $(jq -r .dataset.val_split < $FILE) \
    --test-split $(jq -r .dataset.test_split < $FILE) \
    --image-key $(jq -r .dataset.image_key < $FILE) \
    --label-key $(jq -r .dataset.label_key < $FILE) \
    --head-name $(jq -r .fine_tuning.hparams.head_name < $FILE)
