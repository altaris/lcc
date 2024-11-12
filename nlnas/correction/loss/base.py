"""Abstract loss class"""

from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from numpy.typing import ArrayLike
from torch import Tensor
from torch.utils.data import DataLoader

from ..utils import Matching


class LCCLoss(ABC):
    """Abstract class that encapssulates a loss function for LCC."""

    _tmp: TemporaryDirectory | None = None
    """
    A temporary directory that can be created if needed to save/load data
    before/after broadcast. Cleaned up automatically in
    `on_after_broadcast_cleanup_r0`.
    """

    @abstractmethod
    def __call__(
        self, z: Tensor, y_true: ArrayLike, y_clst: ArrayLike
    ) -> Tensor:
        pass

    def _get_temporary_dir(self) -> Path:
        """Acquires a temporary directory and returns its path."""
        if self._tmp is None:
            self._tmp = TemporaryDirectory()
        return Path(self._tmp.name)

    def on_after_broadcast(self, **kwargs: Any) -> None:
        """
        Should be called on all ranks after the loss object has been
        broadcasted. You probably want to load data from disk here. Does nothing
        by default.
        """

    def on_after_broadcast_cleanup_r0(self, **kwargs: Any) -> None:
        """
        Should be called on rank 0 only after the loss object has been
        broadcasted. There should be a barrier before calling this to make sure
        all ranks finished `on_after_broadcast`. Does nothing by default.
        """
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None

    def on_before_broadcast(self, **kwargs: Any) -> None:
        """
        Should be called on all ranks before broadcasting the loss object from
        rank 0 to other ranks. Does nothing by default.
        """

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
        the begining of each epoch where LCC is to be applied. Also presumably
        called from rank 0 only, and then broadcasted. This implies that the
        loss object should be pickle-able.
        """
