"""A custom dataloader that does things I want."""

from typing import Any, Callable, Iterator

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from ..utils import to_tensor


class EMETDDataLoader(DataLoader):
    """
    A dataloader that does five things:
    1. **E**xtracts: If the underlying dataset is a dataset of dicts of tensors,
       it extracts the tensors at a given key;
    2. **M**asks: Applies a boolean mask over the samples;
    3. **E**venizes: Merges masked batches to evenize the number of samples in
       each batch;
    4. **T**ransforms: Applies a transformation to the batches.
    5. Move to a **D**evice
    """

    device: Any
    key: str | None
    mask: Tensor | None
    transform: Callable[[Tensor], Tensor]
    kwargs: dict

    def __init__(
        self,
        dataset: Dataset,
        key: str | None = None,
        mask: np.ndarray | Tensor | None = None,
        transform: Callable[[Tensor], Tensor] | None = None,
        device: Any = None,
        **kwargs: Any,
    ):
        """
        Args:
            dataset (DataLoader): The underlying dataset.
            key (str | None, optional): The key to use to extract the data from the
                dataloader batches. If left to `None`, batches are assumed to be
                tensors. Otherwise, they are assumed to be dictionaries and the
                actual tensor is located at that key. Defaults to `None`.
            mask (np.ndarray | Tensor | None, optional): A boolean mask of shape
                `(N,)`, where $N$ is at least the length of the dataset. If
                `None`, no masking is applied. Defaults to `None`.
            transform (Callable[[Tensor], Tensor] | None, optional): A function
                to apply to the batches.
            device (Any, optional):
            kwargs: Passed to the DataLoader constructor, see
                https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader.
        """
        super().__init__(dataset, **kwargs)
        self.mask = None if mask is None else to_tensor(mask).bool()
        self.device, self.key = device or "cpu", key
        self.transform, self.kwargs = transform or (lambda x: x), kwargs

    def add_mask(self, mask: np.ndarray | Tensor) -> "EMETDDataLoader":
        """
        Returns a new `EMETDDataLoader` whose mask is the conjunction of the
        current loader's mask and the provided mask
        """
        mask = to_tensor(mask).bool()
        return EMETDDataLoader(
            self.dataset,
            key=self.key,
            mask=mask if self.mask is None else (self.mask & mask),
            transform=self.transform,
            device=self.device,
            **self.kwargs,
        )

    def __iter__(self) -> Iterator[Tensor]:  # type: ignore
        return emetd(
            super().__iter__(),
            key=self.key,
            mask=self.mask,
            batch_size=self.batch_size or 256,
            transform=self.transform,
            device=self.device,
            drop_last=self.drop_last,
        )


def emetd(
    it: Iterator[Tensor] | Iterator[dict[str, Tensor]],
    key: str | None = None,
    mask: np.ndarray | Tensor | None = None,
    batch_size: int = 256,
    transform: Callable[[Tensor], Tensor] | None = None,
    device: Any = None,
    drop_last: bool = False,
) -> Iterator[Tensor]:
    """
    Similar to what `EMETDDataLoader` does, but wraps an iterator directly.
    """

    def _bs() -> int:  # shorthand  # TODO: just keep track of size myself
        return sum(map(len, buffer))

    n_seen, buffer = 0, []
    if transform is None:
        transform = lambda x: x
    for batch in it:
        if key:
            assert isinstance(batch, dict)
            batch = batch[key]
        assert isinstance(batch, Tensor)
        if mask is not None:
            m = mask[n_seen : n_seen + batch.shape[0]]
            n_seen += batch.shape[0]
            batch = batch[m]
        buffer.append(batch)
        if _bs() >= batch_size:
            x = torch.cat(buffer, dim=0)
            yield transform(x[:batch_size]).to(device)
            buffer = [x[batch_size:]] if batch_size < len(x) else []
    if not _bs():
        return  # buffer empty
    if _bs() < batch_size and drop_last:
        return  # last batch too small
    yield transform(torch.cat(buffer, dim=0)).to(device)
