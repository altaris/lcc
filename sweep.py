"""LCC hyperparameters sweep."""

import hashlib
import json
import os
import subprocess
import sys
import warnings
from datetime import datetime
from itertools import product
from pathlib import Path
from time import sleep

import torch
import turbo_broccoli as tb
from loguru import logger as logging

OUTPUT_DIR = Path("out") / "sweep"

DATASETS = [
    {  # https://huggingface.co/datasets/uoft-cs/cifar100
        "name": "cifar100",
        "train_split": "train[:80%]",
        "val_split": "train[80%:]",
        "test_split": "test",
        "image_key": "img",
        "label_key": "fine_label",
    },
    # {  # https://huggingface.co/datasets/timm/imagenet-1k-wds
    #     "name": "timm/imagenet-1k-wds",
    #     "train_split": "train",
    #     "val_split": "validation",
    #     "test_split": "validation",
    #     "image_key": "jpg",
    #     "label_key": "cls",
    # },
]

MODELS = [
    {
        "name": "timm/mobilenetv3_small_050.lamb_in1k",
        "head_name": "classifier",
        "lcc_submodules": ["classifier"],
    },
    {
        "name": "timm/mobilenetv3_small_050.lamb_in1k",
        "head_name": "classifier",
        "lcc_submodules": ["conv_head"],
    },
    {
        "name": "timm/mobilenetv3_small_050.lamb_in1k",
        "head_name": "classifier",
        "lcc_submodules": ["conv_head", "classifier"],
    },
    {
        "name": "timm/resnet18.a3_in1k",
        "head_name": "fc",
        "lcc_submodules": ["fc"],
    },
    {
        "name": "timm/resnet18.a3_in1k",
        "head_name": "fc",
        "lcc_submodules": ["layer4"],
    },
    {
        "name": "timm/resnet18.a3_in1k",
        "head_name": "fc",
        "lcc_submodules": ["layer4", "fc"],
    },
]

LCC_WEIGHTS = [0, 1e-2, 1e-4]
LCC_INTERVALS = [1, 5]
LCC_WARMUPS = [1, 5]
LCC_KS = [2, 5, 10, 50, 100]
SEEDS = [0, 1, 2]

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
    """Setup custom logging format."""
    logging.remove()
    logging.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
            + "[<level>{level: <8}</level>] "
            + (
                "(<blue>{extra[model_name]} {extra[dataset_name]} "
                "sm={extra[lcc_submodules]} "
                "w={extra[lcc_weight]} "
                "k={extra[lcc_k]} "
                # "ccspc={extra[lcc_ccspc]} "
                "itv={extra[lcc_interval]} "
                "wmp={extra[lcc_warmup]}</blue>) "
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
    test_split: str,
    image_key: str,
    label_key: str,
    head_name: str | None,
    seed: int,
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
        "test_split": test_split,
        "image_key": image_key,
        "label_key": label_key,
        "head_name": head_name,
        "seed": seed,
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
        cmd = ["uv", "run", "python", "-m", "nlnas"]
        cmd += ["train", model_name, dataset_name, OUTPUT_DIR]
        cmd += ["--train-split", train_split]
        cmd += ["--val-split", val_split]
        cmd += ["--test-split", test_split]
        cmd += ["--image-key", image_key]
        cmd += ["--label-key", label_key]
        cmd += ["--head-name", head_name]
        cmd += ["--batch-size", 256]
        cmd += ["--max-epochs", 128]
        cmd += ["--seed", seed]
        if lcc_submodules:
            cmd += ["--lcc-submodules", ",".join(lcc_submodules)]
        if lcc_kwargs:
            cmd += ["--lcc-weight", lcc_kwargs["weight"]]
            cmd += ["--lcc-interval", lcc_kwargs["interval"]]
            cmd += ["--lcc-warmup", lcc_kwargs["warmup"]]
            cmd += ["--lcc-k", lcc_kwargs["k"]]
            # cmd += ["--lcc-ccspc", lcc_kwargs["ccspc"]]
        cmd = list(map(str, cmd))
        logging.debug("Spawning subprocess: {}", " ".join(cmd))
        process = subprocess.Popen(cmd)
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"LCC failed (code {process.returncode})")
        tb.save(
            {
                "hostname": os.uname().nodename,
                "start": datetime.now(),
                "conf": cfg,
            },
            OUTPUT_DIR / f"{cfg_hash}.done",
        )
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error("Error: {}", e)
        # raise
    finally:
        logging.debug("Removing lock file {}", lock_file)
        lock_file.unlink()
    logging.debug("Sleeping for 5 seconds in case you want to abort the sweep")
    sleep(5)


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
            LCC_KS,
            SEEDS,
        )
        for (
            lcc_weight,
            lcc_interval,
            lcc_warmup,
            lcc_k,
            seed,
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
                lcc_submodules=(
                    ",".join(lcc_submodules) if lcc_submodules else "/"
                ),
                lcc_weight=lcc_weight if do_lcc else "/",
                lcc_interval=lcc_interval if do_lcc else "/",
                lcc_warmup=lcc_warmup if do_lcc else "/",
                lcc_k=lcc_k if do_lcc else "/",
                # lcc_ccspc=lcc_ccspc if do_lcc else "/",
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
                            "k": lcc_k,
                            # "ccspc": lcc_ccspc,
                        }
                        if do_lcc
                        else None
                    ),
                    train_split=dataset_config["train_split"],
                    val_split=dataset_config["val_split"],
                    test_split=dataset_config["test_split"],
                    image_key=dataset_config["image_key"],
                    label_key=dataset_config["label_key"],
                    head_name=model_config["head_name"],
                    seed=seed,
                )
