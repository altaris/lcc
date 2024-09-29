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

from nlnas.classifiers import BaseClassifier
from nlnas.classifiers.base import full_dataset_latent_clustering
from nlnas.classifiers.huggingface import HuggingFaceClassifier
from nlnas.classifiers.timm import TimmClassifier
from nlnas.datasets import HuggingFaceDataset

# Only need to change the following few variables -----------------------------

# HF_DATASET_NAME = "ILSVRC/imagenet-1k"
HF_DATASET_NAME = "cifar100"
HF_MODEL_NAME = "timm/mobilenetv3_small_050.lamb_in1k"
VERSION = 0

CLUSTERING_METHOD = "louvain"
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


# Automatically set -----------------------------------------------------------

DATASET_NAME = HF_DATASET_NAME.replace("/", "-")
MODEL_NAME = HF_MODEL_NAME.replace("/", "-")
RESULTS_FILE_PATH = (
    Path("out")
    / "ftlcc"
    / DATASET_NAME
    / MODEL_NAME
    / f"results.{VERSION}.json"
)
RESULTS = tb.load(RESULTS_FILE_PATH)

TRAIN_SPLIT = RESULTS["dataset"]["train_split"]
VAL_SPLIT = RESULTS["dataset"]["val_split"]
TEST_SPLIT = RESULTS["dataset"]["test_split"]
IMAGE_KEY = RESULTS["dataset"]["image_key"]
LABEL_KEY = RESULTS["dataset"]["label_key"]

LOGIT_KEY = RESULTS["model"]["hparams"]["logit_key"]
HEAD_NAME = RESULTS["model"]["hparams"]["head_name"]
# if RESULTS["model"]["hparams"].get("lcc_submodules"):
#     LCC_SUBMODULES = RESULTS["model"]["hparams"]["lcc_submodules"]
#     logging.info("Overriding LCC submodules with {}", LCC_SUBMODULES)

CKPT_PATH = (
    Path("out")
    / RESULTS_FILE_PATH.parts[1]
    / RESULTS["model"]["best_checkpoint"]["path"]
)

OUTPUT_DIR = (
    RESULTS_FILE_PATH.parent
    / "analysis"
    / str(RESULTS["model"]["best_checkpoint"]["version"])
    / str(RESULTS["model"]["best_checkpoint"]["epoch"])
)

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
        # lcc_weight=1,  # not actually used, just need a nonzero value,
    )
    model.hparams["lcc_submodules"] = LCC_SUBMODULES
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
        # classes=list(range(100)),
        tqdm_style="console",
    )
    data = {sm: (d.y_clst, d.matching) for sm, d in data.items()}

    (OUTPUT_DIR / CLUSTERING_METHOD).mkdir(parents=True, exist_ok=True)
    tb.save_json(data, OUTPUT_DIR / CLUSTERING_METHOD / "data.json")
