#!/bin/sh

# https://www.patorjk.com/software/taag/
# Font Name: ANSI Regular
echo
echo '███████ ██ ███    ██ ███████     ████████ ██    ██ ███    ██ ██ ███    ██  ██████'
echo '██      ██ ████   ██ ██             ██    ██    ██ ████   ██ ██ ████   ██ ██'
echo '█████   ██ ██ ██  ██ █████          ██    ██    ██ ██ ██  ██ ██ ██ ██  ██ ██   ███'
echo '██      ██ ██  ██ ██ ██             ██    ██    ██ ██  ██ ██ ██ ██  ██ ██ ██    ██'
echo '██      ██ ██   ████ ███████        ██     ██████  ██   ████ ██ ██   ████  ██████'


# MODEL="google/mobilenet_v2_1.0_224"
# HEAD_NAME="classifier"

# MODEL="google/vit-base-patch16-224"
# HEAD_NAME="classifier"

# MODEL="microsoft/resnet-18"
# HEAD_NAME="classifier.1"

# MODEL="timm/mobilenetv3_small_050.lamb_in1k"
# HEAD_NAME="classifier"

MODEL="timm/tinynet_e.in1k"
HEAD_NAME="classifier"

DATASET="cifar100"
TRAIN_SPLIT='train[:80%]'
VAL_SPLIT='train[80%:]'
TEST_SPLIT="test"
IMAGE_KEY="img"
LABEL_KEY="fine_label"

# DATASET="ILSVRC/imagenet-1k"
# TRAIN_SPLIT='train[:80%]'
# VAL_SPLIT='train[80%:]'
# TEST_SPLIT="validation"
# IMAGE_KEY="image"
# LABEL_KEY="label"

echo
echo "=================================================="
echo "MODEL:       $MODEL"
echo "HEAD_NAME:   $HEAD_NAME"
echo "--------------------------------------------------"
echo "DATASET:     $DATASET"
echo "TRAIN_SPLIT: $TRAIN_SPLIT"
echo "VAL_SPLIT:   $VAL_SPLIT"
echo "TEST_SPLIT:  $TEST_SPLIT"
echo "IMAGE_KEY:   $IMAGE_KEY"
echo "LABEL_KEY:   $LABEL_KEY"
echo "=================================================="
echo

python -m nlnas finetune \
    "$MODEL" "$DATASET" out/ft \
    --train-split "$TRAIN_SPLIT" \
    --val-split "$VAL_SPLIT" \
    --test-split "$TEST_SPLIT" \
    --image-key "$IMAGE_KEY" \
    --label-key "$LABEL_KEY" \
    --head-name "$HEAD_NAME"