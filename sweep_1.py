"""LCC hyperparameters sweep."""

import hashlib
import json
import warnings
from itertools import product
from pathlib import Path

import torch
import turbo_broccoli as tb
from loguru import logger as logging

from nlnas.correction.choice import LCCClassSelection
from nlnas.training import train as _train

OUTPUT_DIR = Path("out") / "sweep_1"

DATASETS = [
    {
        "name": "cifar100",
        "train_split": "train[:80%]",
        "val_split": "train[80%:]",
        "image_key": "img",
        "label_key": "fine_label",
    }
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
    #     "name": "microsoft/resnet-18",
    #     "head_name": "classifier.1",
    #     "lcc_submodules": [
    #         "classifier",
    #     ],
    # },
    # {
    #     "name": "microsoft/resnet-18",
    #     "head_name": "classifier.1",
    #     "lcc_submodules": [
    #         "resnet.encoder.stages.3",
    #     ],
    # },
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
]

LCC_WEIGHTS = [1, 1e-2, 1e-4]
LCC_INTERVALS = [1, 2]
LCC_WARMUPS = [0, 1]
LCC_CLASS_SELECTIONS = [None]

STUPID_CUDA_SPAM = r"CUDA call.*failed with initialization error"


def _hash_dict(d: dict) -> str:
    """
    Quick and dirty way to get a unique hash for a (potentially nested)
    dictionary.
    """
    h = hashlib.sha1()
    h.update(json.dumps(d, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def train(
    model_name: str,
    dataset_name: str,
    lcc_submodules: list[str] | None,
    lcc_weight: float | None,
    lcc_interval: int | None,
    lcc_warmup: int | None,
    lcc_class_selection: LCCClassSelection | None,
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

    # Set all LCC hyperparameters to None if we're not actually doing LCC
    do_lcc = (
        (lcc_weight or 0) > 0 and (lcc_interval or 0) > 0 and lcc_submodules
    )
    lcc_submodules = lcc_submodules if do_lcc else None
    lcc_weight = lcc_weight if do_lcc else None
    lcc_interval = lcc_interval if do_lcc else None
    lcc_warmup = lcc_warmup if do_lcc else None
    lcc_class_selection = lcc_class_selection if do_lcc else None

    if do_lcc:
        logging_str = (
            f"{model_name}, {dataset_name}, sms={lcc_submodules}, w={lcc_weight}, "
            f"inter={lcc_interval}, wmp={lcc_warmup}"
        )
    else:
        logging_str = f"{model_name}, {dataset_name}, baseline"

    cfg = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "lcc_submodules": lcc_submodules,
        "lcc_weight": lcc_weight,
        "lcc_interval": lcc_interval,
        "lcc_warmup": lcc_warmup,
        "lcc_class_selection": lcc_class_selection,
        "train_split": train_split,
        "val_split": val_split,
        "image_key": image_key,
        "label_key": label_key,
        "head_name": head_name,
    }
    cfg_hash = _hash_dict(cfg)

    done_file = OUTPUT_DIR / f"{cfg_hash}.done"
    if done_file.exists():
        logging.info("({}): Already trained, skipping", logging_str)
        return

    lock_file = OUTPUT_DIR / f"{cfg_hash}.lock"
    try:
        lock_file.touch(exist_ok=False)
        tb.save_json(cfg, lock_file)
    except FileExistsError:
        logging.info("({}): Being trained, skipping", logging_str)
        return

    try:
        logging.info("({}): Starting training", logging_str)
        train_results = _train(
            model_name=model_name,
            dataset_name=dataset_name,
            output_dir=OUTPUT_DIR,
            lcc_submodules=lcc_submodules,
            lcc_weight=lcc_weight,
            lcc_interval=lcc_interval,
            lcc_warmup=lcc_warmup,
            lcc_class_selection=lcc_class_selection,
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
        logging.error("({}): Error: {}", logging_str, e)
        raise
    finally:
        logging.debug("({}) Removing lock file {}", logging_str, lock_file)
        lock_file.unlink()


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
    warnings.filterwarnings("ignore", message=STUPID_CUDA_SPAM)

    for dataset_config, model_config in product(DATASETS, MODELS):
        everything = (
            [(None, None, None, None)]  # Baseline
            + list(
                product(
                    LCC_WEIGHTS,
                    LCC_INTERVALS,
                    LCC_WARMUPS,
                    LCC_CLASS_SELECTIONS,
                )
            )
        )
        for (
            lcc_weight,
            lcc_interval,
            lcc_warmup,
            lcc_class_selection,
        ) in everything:
            train(
                model_name=model_config["name"],
                dataset_name=dataset_config["name"],
                lcc_submodules=model_config["lcc_submodules"],
                lcc_weight=lcc_weight,
                lcc_interval=lcc_interval,
                lcc_warmup=lcc_warmup,
                lcc_class_selection=lcc_class_selection,
                train_split=dataset_config["train_split"],
                val_split=dataset_config["val_split"],
                image_key=dataset_config["image_key"],
                label_key=dataset_config["label_key"],
                head_name=model_config["head_name"],
            )
