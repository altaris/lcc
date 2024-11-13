#!/bin/sh

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

# MODEL="google/mobilenet_v2_1.0_224"
# HEAD_NAME="classifier"
# LCC_SUBMODULES="classifier"

# MODEL="google/vit-base-patch16-224"
# HEAD_NAME="classifier"
# LCC_SUBMODULES="classifier"

# MODEL="microsoft/resnet-18"
# HEAD_NAME="classifier.1"
# LCC_SUBMODULES="classifier"

MODEL="timm/resnet18.a3_in1k"
HEAD_NAME="fc"
LCC_SUBMODULES="fc"

# MODEL="timm/mobilenetv3_small_050.lamb_in1k"
# HEAD_NAME="classifier"
# LCC_SUBMODULES="conv_head,classifier"

# MODEL="timm/tinynet_e.in1k"
# HEAD_NAME="classifier"
# LCC_SUBMODULES="conv_head"

CE_WEIGHT=1
LCC_INTERVAL=5
LCC_K=2
LCC_WARMUP=1
LCC_WEIGHT=1e-2

OUTPUT_DIR="out.test"
export CUDA_VISIBLE_DEVICES=0,1


# https://www.patorjk.com/software/taag/
# Font Name: ANSI Regular
echo
echo "                     ██       ██████  ██████ "
echo "                     ██      ██      ██      "
echo "                     ██      ██      ██      "
echo "                     ██      ██      ██      "
echo "                     ███████  ██████  ██████ "
echo
echo "======================================================================"
echo "OUTPUT_DIR:     $OUTPUT_DIR"
echo "Cuda devs:      $CUDA_VISIBLE_DEVICES"
echo "----------------------------------------------------------------------"
echo "MODEL:          $MODEL"
echo "HEAD_NAME:      $HEAD_NAME"
echo "----------------------------------------------------------------------"
echo "CE_WEIGHT:      $CE_WEIGHT"
echo "LCC_INTERVAL:   $LCC_INTERVAL"
echo "LCC_K:          $LCC_K"
echo "LCC_SUBMODULES: $LCC_SUBMODULES"
echo "LCC_WARMUP:     $LCC_WARMUP"
echo "LCC_WEIGHT:     $LCC_WEIGHT"
echo "----------------------------------------------------------------------"
echo "DATASET:        $DATASET"
echo "TRAIN_SPLIT:    $TRAIN_SPLIT"
echo "VAL_SPLIT:      $VAL_SPLIT"
echo "TEST_SPLIT:     $TEST_SPLIT"
echo "IMAGE_KEY:      $IMAGE_KEY"
echo "LABEL_KEY:      $LABEL_KEY"
echo "======================================================================"
echo

uv run python -m nlnas train \
    "$MODEL" \
    "$DATASET" \
    "$OUTPUT_DIR" \
    --ce-weight "$CE_WEIGHT" \
    --lcc-submodules "$LCC_SUBMODULES" \
    --lcc-weight "$LCC_WEIGHT" \
    --lcc-interval "$LCC_INTERVAL" \
    --batch-size 256 \
    --lcc-warmup "$LCC_WARMUP" \
    --lcc-k "$LCC_K" \
    --train-split "$TRAIN_SPLIT" \
    --val-split "$VAL_SPLIT" \
    --test-split "$TEST_SPLIT" \
    --image-key "$IMAGE_KEY" \
    --label-key "$LABEL_KEY" \
    --head-name "$HEAD_NAME"
