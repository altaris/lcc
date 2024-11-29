"""Abstract loss class"""

from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from numpy.typing import ArrayLike
from pytorch_lightning.strategies import ParallelStrategy, Strategy
from torch import Tensor
from torch.utils.data import DataLoader

from ..utils import Matching


class LCCLoss(ABC):
    """Abstract class that encapssulates a loss function for LCC."""

    strategy: ParallelStrategy | None

    _tmp_dir: TemporaryDirectory | None = None
    """
    A temporary directory that can be created if needed to save/load data
    before/after sync. Cleaned up automatically in
    `on_after_sync`.
    """

    @abstractmethod
    def __call__(
        self, z: Tensor, y_true: ArrayLike, y_clst: ArrayLike
    ) -> Tensor:
        pass

    def __init__(self, strategy: Strategy | None = None) -> None:
        # TODO: Log a warning if strategy is a Strategy but not a
        # ParallelStrategy
        self.strategy = (
            strategy if isinstance(strategy, ParallelStrategy) else None
        )

    def _get_tmp_dir(self) -> Path:
        """
        On rank 0, acquires a temporary directory, broadcast the handler to all
        ranks, and returns its path.

        Warning:
            In a distributed environment, this method must be called from all
            ranks "at the same time".

        Args:
            broadcast (bool, optional):
        """
        if self._tmp_dir is not None:
            return Path(self._tmp_dir.name)
        if self.strategy is None:
            self._tmp_dir = TemporaryDirectory()
            return Path(self._tmp_dir.name)
        if self.strategy.global_rank == 0:
            self._tmp_dir = TemporaryDirectory()
        else:
            self._tmp_dir = None
        self._tmp_dir = self.strategy.broadcast(self._tmp_dir, src=0)
        assert self._tmp_dir is not None
        return Path(self._tmp_dir.name)

    def on_after_sync(self, **kwargs: Any) -> None:
        """
        Should be called on all ranks after the loss object has been synced.
        """
        if self._tmp_dir is not None and (
            self.strategy is None or self.strategy.global_rank == 0
        ):
            self._tmp_dir.cleanup()

    def on_before_sync(self, **kwargs: Any) -> None:
        """
        Should be called on all ranks before syncing the loss object from rank 0
        to other ranks.
        """

    def sync(self, **kwargs: Any) -> None:
        """
        Distributes or shares this object's data across all ranks. Call this
        after each ranks called `update`. Is just a barrier by default.
        """
        if self.strategy is not None:
            self.strategy.barrier()

    @abstractmethod
    def update(
        self,
        dl: DataLoader,
        y_true: ArrayLike,
        y_clst: ArrayLike,
        matching: Matching,
    ) -> None:
        """
        Updates the internal state of the loss function. Presumably called at
        the begining of each epoch where LCC is to be applied.

        The dataloader has to yield batches that are tuples of tensors, the
        first of which is a 2D tensor of samples.

        Warning:
            If the construction of the loss object is distributed across
            multiple ranks, make sure that `dl` iterate over the WHOLE dataset
            (no distributed sampling).
        """
