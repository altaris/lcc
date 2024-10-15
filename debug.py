import torch

from nlnas.training import train

DATASET = "cifar100"
TRAIN_SPLIT = "train[:20%]"
VAL_SPLIT = "train[80%:]"
TEST_SPLIT = "test"
IMAGE_KEY = "img"
LABEL_KEY = "fine_label"

MODEL = "timm/mobilenetv3_small_050.lamb_in1k"
HEAD_NAME = "classifier"
LCC_SUBMODULES = ["conv_head", "classifier"]

LCC_WEIGHT = 1e-4
LCC_INTERVAL = 1
LCC_WARMUP = 0
CE_WEIGHT = 1
LCC_K = 5

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_float32_matmul_precision("medium")
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
        max_epochs=10,
        batch_size=128,
        train_split=TRAIN_SPLIT,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT,
        image_key=IMAGE_KEY,
        label_key=LABEL_KEY,
        head_name=HEAD_NAME,
    )
