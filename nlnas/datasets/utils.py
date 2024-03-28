"""Various utilities pertaining to datasets"""

import torch
from loguru import logger as logging
from torch import Tensor
from torch.utils.data import DataLoader


def dl_head(dl: DataLoader, n: int) -> list[Tensor]:
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


def dl_targets(dl: DataLoader) -> set:
    """
    Returns (distinct) targets of the dataset underlying this dataloader. Has
    to iterate through the whole dataset, so it can be horribly inefficient =(
    """
    tgts = set()
    for batch in iter(dl):
        tgts.update(batch[-1].tolist())
    return tgts
