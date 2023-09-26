"""Useful stuff"""

import os

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from loguru import logger as logging


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
