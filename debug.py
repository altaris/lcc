import torch
from loguru import logger as logging

from lcc.logging import setup_logging
from lcc.training import train

DATASET = "cifar100"
TRAIN_SPLIT = "train[:20%]"
VAL_SPLIT = "train[80%:]"
TEST_SPLIT = "test"
IMAGE_KEY = "img"
LABEL_KEY = "fine_label"

# DATASET = "ILSVRC/imagenet-1k"
# TRAIN_SPLIT = "train[:1%]"
# VAL_SPLIT = "train[99%:]"
# TEST_SPLIT = VAL_SPLIT
# IMAGE_KEY = "image"
# LABEL_KEY = "label"

# DATASET = "timm/imagenet-1k-wds"
# TRAIN_SPLIT = "train[:1%]"
# VAL_SPLIT = "validation[99%:]"
# TEST_SPLIT = VAL_SPLIT
# IMAGE_KEY = "jpg"
# LABEL_KEY = "cls"

# MODEL = "timm/mobilenetv3_small_050.lamb_in1k"
# HEAD_NAME = "classifier"
# LCC_SUBMODULES = ["conv_head", "classifier"]
# LOGIT_KEY = "logits"

MODEL = "alexnet"
HEAD_NAME = "classifier.6"
LCC_SUBMODULES = ["classifier.4"]
LOGIT_KEY = None

LCC_WEIGHT = 1e-4
LCC_INTERVAL = 1
LCC_WARMUP = 0
CE_WEIGHT = 1
LCC_K = 50

if __name__ == "__main__":
    # torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_float32_matmul_precision("medium")
    setup_logging(logging_level="DEBUG")
    try:
        train(
            model_name=MODEL,
            dataset_name=DATASET,
            output_dir="out.test",
            ce_weight=CE_WEIGHT,
            lcc_submodules=LCC_SUBMODULES,
            lcc_kwargs={
                "interval": LCC_INTERVAL,
                "warmup": LCC_WARMUP,
                "weight": LCC_WEIGHT,
                "k": LCC_K,
            },
            max_epochs=2,
            batch_size=128,
            train_split=TRAIN_SPLIT,
            val_split=VAL_SPLIT,
            test_split=TEST_SPLIT,
            image_key=IMAGE_KEY,
            label_key=LABEL_KEY,
            logit_key=LOGIT_KEY,
            head_name=HEAD_NAME,
        )
    except:
        logging.exception("WTF")
        raise
