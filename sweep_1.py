"""LCC hyperparameters sweep."""

import hashlib
import json
import os
import sys
import warnings
from datetime import datetime
from itertools import product
from pathlib import Path

import torch
import turbo_broccoli as tb
from loguru import logger as logging

from nlnas.correction.choice import LCCClassSelection
from nlnas.training import train as _train

OUTPUT_DIR = Path("out") / "sweep_1"

DATASETS = [
    {  # https://huggingface.co/datasets/uoft-cs/cifar100
        "name": "cifar100",
        "train_split": "train[:80%]",
        "val_split": "train[80%:]",
        "test_split": "test",
        "image_key": "img",
        "label_key": "fine_label",
    },
    # {  # https://huggingface.co/datasets/timm/eurosat-rgb
    #     "name": "timm/eurosat-rgb",
    #     "train_split": "train",
    #     "val_split": "validation",
    #     "test_split": "test",
    #     "image_key": "image",
    #     "label_key": "label",
    # },
    # {  # https://huggingface.co/datasets/timm/resisc45
    #     "name": "timm/resisc45rgb",
    #     "train_split": "train",
    #     "val_split": "validation",
    #     "test_split": "test",
    #     "image_key": "image",
    #     "label_key": "label",
    # },
]

MODELS = [
    {
        "name": "google/mobilenet_v2_1.0_224",
        "head_name": "classifier",
        "lcc_submodules": [
            "classifier",
        ],
    },
    {
        "name": "google/mobilenet_v2_1.0_224",
        "head_name": "classifier",
        "lcc_submodules": [
            "mobilenet_v2.conv_1x1",
        ],
    },
    # {
    #     "name": "google/mobilenet_v2_1.0_224",
    #     "head_name": "classifier",
    #     "lcc_submodules": [
    #         "mobilenet_v2.conv_1x1",
    #         "classifier",
    #     ],
    # },
    {
        "name": "microsoft/resnet-18",
        "head_name": "classifier.1",
        "lcc_submodules": [
            "classifier",
        ],
    },
    {
        "name": "microsoft/resnet-18",
        "head_name": "classifier.1",
        "lcc_submodules": [
            "resnet.encoder.stages.3",
        ],
    },
    # {
    #     "name": "microsoft/resnet-18",
    #     "head_name": "classifier.1",
    #     "lcc_submodules": [
    #         "resnet.encoder.stages.3",
    #         "classifier",
    #     ],
    # },
    # {
    #     "name": "microsoft/resnet-18",
    #     "head_name": "classifier.1",
    #     "lcc_submodules": [
    #         "resnet.encoder.stages.2",
    #         "resnet.encoder.stages.3",
    #         "classifier",
    #     ],
    # },
    # {
    #     "name": "microsoft/resnet-18",
    #     "head_name": "classifier.1",
    #     "lcc_submodules": [
    #         "resnet.encoder.stages.1",
    #         "resnet.encoder.stages.2",
    #         "resnet.encoder.stages.3",
    #         "classifier",
    #     ],
    # },
    {
        "name": "timm/tinynet_e.in1k",
        "head_name": "classifier",
        "lcc_submodules": [
            "classifier",
        ],
    },
    {
        "name": "timm/tinynet_e.in1k",
        "head_name": "classifier",
        "lcc_submodules": [
            "conv_head",
        ],
    },
    # {
    #     "name": "timm/tinynet_e.in1k",
    #     "head_name": "classifier",
    #     "lcc_submodules": [
    #         "conv_head",
    #         "classifier",
    #     ],
    # },
    {
        "name": "timm/vgg11.tv_in1k",
        "head_name": "head.fc",
        "lcc_submodules": [
            "head",
        ],
    },
    {
        "name": "timm/vgg11.tv_in1k",
        "head_name": "head.fc",
        "lcc_submodules": [
            "pre_logits",
        ],
    },
    # {
    #     "name": "timm/vgg11.tv_in1k",
    #     "head_name": "head.fc",
    #     "lcc_submodules": [
    #         "pre_logits",
    #         "head",
    #     ],
    # },
    {
        "name": "timm/convnext_small.in12k",
        "head_name": "head.fc",
        "lcc_submodules": [
            "head",
        ],
    },
    {
        "name": "timm/convnext_small.in12k",
        "head_name": "head.fc",
        "lcc_submodules": [
            "stages.3",
        ],
    },
    # {
    #     "name": "timm/convnext_small.in12k",
    #     "head_name": "head.fc",
    #     "lcc_submodules": [
    #         "stages.3",
    #         "head",
    #     ],
    # },
    {
        "name": "timm/tf_efficientnet_l2.ns_jft_in1k",
        "head_name": "classifier",
        "lcc_submodules": [
            "classifier",
        ],
    },
    {
        "name": "timm/tf_efficientnet_l2.ns_jft_in1k",
        "head_name": "classifier",
        "lcc_submodules": [
            "conv_head",
        ],
    },
    # {
    #     "name": "timm/tf_efficientnet_l2.ns_jft_in1k",
    #     "head_name": "classifier",
    #     "lcc_submodules": [
    #         "conv_head",
    #         "classifier",
    #     ],
    # },
]

LCC_WEIGHTS = [0, 1, 1e-2, 1e-4]
LCC_INTERVALS = [1, 5]
LCC_WARMUPS = [1]
LCC_CLASS_SELECTIONS = [None, "max_connected"]

STUPID_CUDA_SPAM = r"CUDA call.*failed with initialization error"


def _hash_dict(d: dict) -> str:
    """
    Quick and dirty way to get a unique hash for a (potentially nested)
    dictionary.
    """
    h = hashlib.sha1()
    h.update(json.dumps(d, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def setup_logging(logging_level: str = "debug") -> None:
    """
    Sets logging format and level. The format is

        %(asctime)s [%(levelname)-8s] %(message)s

    e.g.

        2022-02-01 10:41:43,797 [INFO    ] Hello world
        2022-02-01 10:42:12,488 [CRITICAL] We're out of beans!

    Args:
        logging_level (str): Logging level in `LOGGING_LEVELS` (case
            insensitive).
    """
    logging.remove()
    logging.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
            + "[<level>{level: <8}</level>] "
            + (
                "(<blue>{extra[model_name]} {extra[dataset_name]} "
                "sm={extra[lcc_submodules]} w={extra[lcc_weight]} "
                "itv={extra[lcc_interval]} wmp={extra[lcc_warmup]}</blue>) "
            )
            + "<level>{message}</level>"
        ),
        level=logging_level.upper(),
        enqueue=True,
        colorize=True,
    )


def train(
    model_name: str,
    dataset_name: str,
    lcc_submodules: list[str] | None,
    lcc_kwargs: dict | None,
    train_split: str,
    val_split: str,
    image_key: str,
    label_key: str,
    head_name: str | None,
) -> None:
    """
    Train a model if it hasn't been trained yet. The hash of the configuration
    is used to determine if the model has been trained.
    """

    cfg = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "lcc_submodules": lcc_submodules,
        "lcc_kwargs": lcc_kwargs,
        "train_split": train_split,
        "val_split": val_split,
        "image_key": image_key,
        "label_key": label_key,
        "head_name": head_name,
    }
    cfg_hash = _hash_dict(cfg)

    done_file = OUTPUT_DIR / f"{cfg_hash}.done"
    if done_file.exists():
        logging.info("Already trained, skipping")
        return

    lock_file = OUTPUT_DIR / f"{cfg_hash}.lock"
    try:
        lock_file.touch(exist_ok=False)
        tb.save_json(
            {
                "hostname": os.uname().nodename,
                "start": datetime.now(),
                "conf": cfg,
            },
            lock_file,
        )
    except FileExistsError:
        logging.info("Being trained, skipping")
        return

    try:
        logging.info("Starting training")
        logging.debug("Lock file: {}", lock_file)
        train_results = _train(
            model_name=model_name,
            dataset_name=dataset_name,
            output_dir=OUTPUT_DIR,
            lcc_submodules=lcc_submodules,
            lcc_kwargs=lcc_kwargs,
            train_split=train_split,
            val_split=val_split,
            image_key=image_key,
            label_key=label_key,
            head_name=head_name,
        )
        tb.save(train_results, OUTPUT_DIR / f"{cfg_hash}.done")
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error("Error: {}", e)
        # raise
    finally:
        logging.debug("Removing lock file {}", lock_file)
        lock_file.unlink()


if __name__ == "__main__":
    setup_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
    warnings.filterwarnings("ignore", message=STUPID_CUDA_SPAM)

    for dataset_config, model_config in product(DATASETS, MODELS):
        everything = product(
            LCC_WEIGHTS,
            LCC_INTERVALS,
            LCC_WARMUPS,
            LCC_CLASS_SELECTIONS,
        )
        for (
            lcc_weight,
            lcc_interval,
            lcc_warmup,
            lcc_class_selection,
        ) in everything:
            lcc_submodules = model_config["lcc_submodules"]
            do_lcc = (
                (lcc_weight or 0) > 0
                and (lcc_interval or 0) > 0
                and lcc_submodules
            )
            lcc_submodules = lcc_submodules if do_lcc else None
            with logging.contextualize(
                model_name=model_config["name"],
                dataset_name=dataset_config["name"],
                lcc_submodules=lcc_submodules,
                lcc_weight=lcc_weight,
                lcc_interval=lcc_interval,
                lcc_warmup=lcc_warmup,
            ):
                train(
                    model_name=model_config["name"],
                    dataset_name=dataset_config["name"],
                    lcc_submodules=lcc_submodules,
                    lcc_kwargs=(
                        {
                            "weight": lcc_weight,
                            "interval": lcc_interval,
                            "warmup": lcc_warmup,
                            "class_selection": lcc_class_selection,
                        }
                        if do_lcc
                        else None
                    ),
                    train_split=dataset_config["train_split"],
                    val_split=dataset_config["val_split"],
                    image_key=dataset_config["image_key"],
                    label_key=dataset_config["label_key"],
                    head_name=model_config["head_name"],
                )
