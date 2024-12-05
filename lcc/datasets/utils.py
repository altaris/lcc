"""Various utilities pertaining to datasets"""

from typing import TypeAlias

import torch
from loguru import logger as logging
from torch import Tensor
from torch.utils.data import DataLoader

Batch: TypeAlias = dict[str, Tensor]


def dl_head(dl: DataLoader, n: int) -> list[Batch]:
    """
    Returns the first `n` samples of a DataLoader **as a list of batches**.

    Args:
        dl (DataLoader):
        n (int):

    Warning:
        Only supports batches that are dicts of tensors.
    """

    def _n() -> int:
        if not batches:
            return 0
        k = list(batches[0].keys())[0]
        return sum(map(lambda b: len(b[k]), batches))

    batches: list[Batch] = []
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
    if (r := _n() - n) > 0:
        for k in batches[-1].keys():
            batches[-1][k] = batches[-1][k][:-r]
    return batches


def flatten_batches(batches: list[Batch]) -> Batch:
    """
    Flattens a list of batches into a single batch.

    Args:
        batches (list[Batch]):
    """
    return {
        k: torch.concat([b[k] for b in batches]) for k in batches[0].keys()
    }
