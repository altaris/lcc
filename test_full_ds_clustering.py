"""
Loads a pretrained model and performs full-dataset latent clustering.
1. The whole dataset is evaluated and the latent representations (for the layers
specified in LCC_SUBMODULES) are saved.
2. For each selected submodule, the latent representations are clustered using
the Louvain method.

See also:
    - `nlnas.utils.pretty_print_submodules` for the list of submodules in a
      model.
    - `nlnas.classifier.base.full_dataset_latent_clustering`
"""

from pathlib import Path

import torch
import turbo_broccoli as tb
from loguru import logger as logging

from nlnas import (
    HuggingFaceClassifier,
    HuggingFaceDataset,
    TimmClassifier,
    full_dataset_latent_clustering,
)
from nlnas.classifiers import BaseClassifier

# Only need to change the 4 following variables -------------------------------

# HF_DATASET_NAME = "ILSVRC/imagenet-1k"
HF_DATASET_NAME = "cifar100"


HF_MODEL_NAME = "timm/mobilenetv3_small_050.lamb_in1k"
LCC_SUBMODULES = [
    # "blocks.0",
    "blocks.1",
    # "blocks.2",
    "blocks.3",
    # "blocks.4",
    "blocks.5",
    "conv_head",
    "classifier",
]


# HF_MODEL_NAME = "microsoft/resnet-18"
# LCC_SUBMODULES = [  # See also `nlnas.utils.pretty_print_submodules`
#     "resnet.encoder.stages.0",
#     "resnet.encoder.stages.1",
#     "resnet.encoder.stages.2",
#     "resnet.encoder.stages.3",
#     "classifier.1",
# ]

# HF_MODEL_NAME = "google/mobilenet_v2_1.0_224"
# LCC_SUBMODULES = [  # See also `nlnas.utils.pretty_print_submodules`
#     "mobilenet_v2.layer.0",
#     # "mobilenet_v2.layer.1",
#     # "mobilenet_v2.layer.2",
#     # "mobilenet_v2.layer.3",
#     # "mobilenet_v2.layer.4",
#     "mobilenet_v2.layer.5",
#     # "mobilenet_v2.layer.6",
#     # "mobilenet_v2.layer.7",
#     # "mobilenet_v2.layer.8",
#     # "mobilenet_v2.layer.9",
#     "mobilenet_v2.layer.10",
#     # "mobilenet_v2.layer.11",
#     # "mobilenet_v2.layer.12",
#     # "mobilenet_v2.layer.13",
#     # "mobilenet_v2.layer.14",
#     "mobilenet_v2.layer.15",
#     # "conv_1x1",
#     "classifier",
# ]


CLUSTERING_METHOD = "louvain"

# Automatically set -----------------------------------------------------------

DATASET_NAME = HF_DATASET_NAME.replace("/", "-")
MODEL_NAME = HF_MODEL_NAME.replace("/", "-")
FT_RESULT_FILE = (
    Path("./out") / "ft" / DATASET_NAME / MODEL_NAME / "results.json"
)
FT_RESULTS = tb.load(FT_RESULT_FILE)

TRAIN_SPLIT = FT_RESULTS["dataset"]["train_split"]
VAL_SPLIT = FT_RESULTS["dataset"]["val_split"]
TEST_SPLIT = FT_RESULTS["dataset"]["test_split"]
IMAGE_KEY = FT_RESULTS["dataset"]["image_key"]
LABEL_KEY = FT_RESULTS["dataset"]["label_key"]

LOGIT_KEY = FT_RESULTS["fine_tuning"]["hparams"]["logit_key"]
HEAD_NAME = FT_RESULTS["fine_tuning"]["hparams"]["head_name"]
CLST_WEIGHT = 1  # just need a nonzero value

CKPT_PATH = (
    Path("./out") / "ft" / FT_RESULTS["fine_tuning"]["best_checkpoint"]["path"]
)

OUTPUT_DIR = Path("out") / "lc" / DATASET_NAME / MODEL_NAME

if __name__ == "__main__":
    (OUTPUT_DIR / CLUSTERING_METHOD).mkdir(parents=True, exist_ok=True)

    ClassifierClass: type[BaseClassifier]
    if HF_MODEL_NAME.startswith("timm/"):
        ClassifierClass = TimmClassifier
    else:
        ClassifierClass = HuggingFaceClassifier

    logging.info("Loading dataset '{}'", HF_DATASET_NAME)
    dataset = HuggingFaceDataset(
        dataset_name=HF_DATASET_NAME,
        fit_split=TRAIN_SPLIT,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT,
        label_key=LABEL_KEY,
        image_processor=ClassifierClass.get_image_processor(HF_MODEL_NAME),
    )
    dataset.setup("fit")

    logging.info(
        "Loading model '{}' with weights from {}", HF_MODEL_NAME, CKPT_PATH
    )
    model = ClassifierClass(
        model_name=HF_MODEL_NAME,
        n_classes=dataset.n_classes(),
        head_name=HEAD_NAME,
        image_key=IMAGE_KEY,
        label_key=LABEL_KEY,
        logit_key=LOGIT_KEY,
        lcc_submodules=LCC_SUBMODULES,
        clst_weight=CLST_WEIGHT,
    )
    # pylint: disable=no-value-for-parameter
    model.model = ClassifierClass.load_from_checkpoint(CKPT_PATH).model
    if torch.cuda.is_available():
        model = model.to("cuda")  # Just for evaluation

    logging.info("Starting full-dataset latent clustering in {}", OUTPUT_DIR)
    data = full_dataset_latent_clustering(
        model=model,
        dataset=dataset,
        output_dir=OUTPUT_DIR / "embeddings",
        method=CLUSTERING_METHOD,
        device="cpu",  # Dataset might not fit in the GPU
        classes=list(range(100)),
        tqdm_style="console",
    )

    tb.save_json(data, OUTPUT_DIR / CLUSTERING_METHOD / "data.json")
