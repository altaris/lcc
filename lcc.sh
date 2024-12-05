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
# LOGIT_KEY="logits"

# MODEL="google/vit-base-patch16-224"
# HEAD_NAME="classifier"
# LCC_SUBMODULES="classifier"
# LOGIT_KEY="logits"

# MODEL="microsoft/resnet-18"
# HEAD_NAME="classifier.1"
# LCC_SUBMODULES="resnet.encoder.stages.3"
# LOGIT_KEY="logits"

# MODEL="timm/resnet18.a3_in1k"
# HEAD_NAME="fc"
# LCC_SUBMODULES="fc"
# LOGIT_KEY="logits"

# MODEL="timm/mobilenetv3_small_050.lamb_in1k"
# HEAD_NAME="classifier"
# LCC_SUBMODULES="conv_head,classifier"
# LOGIT_KEY="logits"

# MODEL="timm/tinynet_e.in1k"
# HEAD_NAME="classifier"
# LCC_SUBMODULES="conv_head"
# LOGIT_KEY="logits"

MODEL="alexnet"
HEAD_NAME="classifier.6"
LCC_SUBMODULES="classifier.4"
LOGIT_KEY=""

CE_WEIGHT=1
LCC_INTERVAL=1
LCC_K=50
LCC_WARMUP=0
LCC_WEIGHT=1e-2

BATCH_SIZE=256
MAX_EPOCHS=1

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
echo "LOGIT_KEY:      $LOGIT_KEY"
echo "----------------------------------------------------------------------"
echo "CE_WEIGHT:      $CE_WEIGHT"
echo "LCC_INTERVAL:   $LCC_INTERVAL"
echo "LCC_K:          $LCC_K"
echo "LCC_SUBMODULES: $LCC_SUBMODULES"
echo "LCC_WARMUP:     $LCC_WARMUP"
echo "LCC_WEIGHT:     $LCC_WEIGHT"
echo "----------------------------------------------------------------------"
echo "BATCH_SIZE:     $BATCH_SIZE"
echo "MAX_EPOCHS:     $MAX_EPOCHS"
echo "----------------------------------------------------------------------"
echo "DATASET:        $DATASET"
echo "TRAIN_SPLIT:    $TRAIN_SPLIT"
echo "VAL_SPLIT:      $VAL_SPLIT"
echo "TEST_SPLIT:     $TEST_SPLIT"
echo "IMAGE_KEY:      $IMAGE_KEY"
echo "LABEL_KEY:      $LABEL_KEY"
echo "======================================================================"
echo

uv run python -m lcc \
    --logging-level "DEBUG" \
    train \
    "$MODEL" \
    "$DATASET" \
    "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --ce-weight "$CE_WEIGHT" \
    --head-name "$HEAD_NAME" \
    --image-key "$IMAGE_KEY" \
    --label-key "$LABEL_KEY" \
    --logit-key "$LOGIT_KEY" \
    --lcc-interval "$LCC_INTERVAL" \
    --lcc-k "$LCC_K" \
    --lcc-submodules "$LCC_SUBMODULES" \
    --lcc-warmup "$LCC_WARMUP" \
    --lcc-weight "$LCC_WEIGHT" \
    --max-epochs "$MAX_EPOCHS" \
    --seed 0 \
    --test-split "$TEST_SPLIT" \
    --train-split "$TRAIN_SPLIT" \
    --val-split "$VAL_SPLIT"
