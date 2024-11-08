"""Abstract loss class"""

from abc import ABC, abstractmethod

from numpy.typing import ArrayLike
from torch import Tensor
from torch.utils.data import DataLoader

from ..utils import Matching


class LCCLoss(ABC):
    """Abstract class that encapssulates a loss function for LCC."""

    @abstractmethod
    def __call__(
        self, z: Tensor, y_true: ArrayLike, y_clst: ArrayLike
    ) -> Tensor:
        pass

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
