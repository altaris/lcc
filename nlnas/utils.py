"""Useful stuff"""

import os
from glob import glob
from pathlib import Path

import regex as re
import torch
from loguru import logger as logging
from torch import Tensor
from torch.utils.data import DataLoader


def all_ckpt_paths(ckpts_dir_path: str | Path) -> list[Path]:
    """
    Returns the sorted (by epoch) list of all checkpoint file paths in a given
    directory. `ckpts_dir_path` probably looks like
    `.../tb_logs/my_model/version_N/checkpoints`. The checkpoint files are
    assumed to be named as `epoch=XX-step=YY.ckpt` where of course `XX` is the
    epoch number and `YY` is the step number.

    Args:
        ckpts_dir_path (str | Path):
    """
    r, d = re.compile(r"/epoch=(\d+)-step=\d+\.ckpt$"), {}
    for p in glob(str(Path(ckpts_dir_path) / "*.ckpt")):
        if m := re.search(r, p):
            epoch = int(m.group(1))
            d[epoch] = Path(p)
    return [d[i] for i in sorted(list(d.keys()))]


def best_device() -> str:
    """Self-explanatory"""
    accelerator = os.getenv("PL_ACCELERATOR", "auto").lower()
    if accelerator == "gpu" and torch.cuda.is_available():
        return "cuda"
    if accelerator == "auto":
        return (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    return accelerator


def get_first_n(dl: DataLoader, n: int) -> list[Tensor]:
    """
    Gets a batch of length `n` consisting of the first `n` samples of the
    dataloader.
    """

    def _n():
        return sum(map(lambda b: len(b[0]), batches))

    batches: list[list[Tensor]] = []
    it = iter(dl)
    try:
        while _n() < n:
            batches.append(next(it))
    except StopIteration:
        logging.warning(
            "Tried to extract {} elements from dataset but only found {}",
            n,
            _n(),
        )
    return list(map(lambda l: torch.concat(l)[:n], zip(*batches)))


def targets(dl: DataLoader) -> set:
    """
    Returns (distinct) targets of the dataset underlying this dataloader. Has
    to iterate through the whole dataset, so it can be horribly inefficient =(
    """
    tgts = set()
    for batch in iter(dl):
        tgts.update(batch[-1].tolist())
    return tgts
