"""Some random utilities"""

from collections.abc import Iterable
from typing import Any, Iterator, TypeAlias

Matching: TypeAlias = (
    dict[int, set[int]]
    | dict[str, set[int]]
    | dict[int, set[str]]
    | dict[str, set[str]]
)


class EarlyStoppingLoop(Iterable):
    """
    A loop that abstract early stopping logics.

    Example:

        ```python
        loop = EarlyStoppingLoop(max_iter=100, patience=10, delta=0.01)
        loop.propose(initial_solution, float("inf"))
        for _ in loop:
            ...
            loop.propose(new_solution, new_score)
        best_solution, best_score = loop.best()
        ```
    """

    max_iter: int
    patience: int
    delta: float

    _best_score: float = float("inf")
    _best_solution: Any = None
    _iteration: int = 0
    _sli: int = 0  # Nb of iterations since last improvement

    def __init__(
        self, max_iter: int = 100, patience: int = 10, delta: float = 0
    ):
        """
        Args:
            max_iter (int, optional):
            patience (int, optional):
            delta (float, optional):
        """
        self.max_iter, self.patience, self.delta = max_iter, patience, delta

    def __iter__(self) -> Iterator:
        self._best_solution, self._best_score = None, float("inf")
        self._iteration, self._sli = 0, 0
        return self

    def __len__(self) -> int:
        return self.max_iter

    def __next__(self) -> int:
        if (self._iteration >= self.max_iter) or (self._sli >= self.patience):
            raise StopIteration
        self._iteration += 1
        return self._iteration - 1

    def best(self) -> tuple[Any, float]:
        """Returns the best solution and its score"""
        return self._best_solution, self._best_score

    def propose(self, solution: Any, score: float) -> None:
        """Proposes a new solution and score"""
        if score < self._best_score - self.delta:
            self._best_solution, self._best_score = solution, score
            self._sli = 0
        else:
            self._sli += 1


def to_int_matching(matching: Matching) -> dict[int, set[int]]:
    """
    Sometimes, when a matching is loaded from a JSON file, the keys are
    strings rather than ints. This function converts the keys to ints.
    """
    return {int(k): set(int(x) for x in v) for k, v in matching.items()}
