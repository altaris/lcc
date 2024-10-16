"""See the `nlnas.datasets.BatchedTensorDataset` class documentation."""

from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np
import torch
from safetensors import torch as st
from torch import Tensor
from torch.utils.data import IterableDataset

from ..utils import make_tqdm, to_tensor
from .emetd_dataloader import EMETDDataLoader


class BatchedTensorDataset(IterableDataset):
    """
    Dataset that load tensor batches produced by
    `nlnas.datasets.save_tensor_batched`.
    """

    key: str
    paths: list[Path]

    def __init__(
        self,
        batch_dir: str | Path,
        prefix: str = "batch",
        extension: str = "st",
        key: str = "",
    ):
        """
        See `nlnas.datasets.save_tensor_batched` for the precise meaning of the
        argument. But in a few words, this dataset will load batches from
        [Safetensors](https://huggingface.co/docs/safetensors/index) files named
        after the following scheme:

            batch_dir/<prefix>.<batch_idx>.<extension>

        Warning:
            The list of batch files will be globed from `batch_dir` upon
            instantiation of this dataset class. It will not be updated
            afterwards.

        Args:
            batch_dir (str | Path):
            prefix (str, optional):
            extension (str, optional): Without the first `.`. Defaults to `st`.
            key (str, optional): The key to use when saving the file. Batches
                are stored in safetensor files, which are essentially
                dictionaries.  This arg specifies which key contains the data of
                interest.
        """
        self.paths = list(
            sorted(Path(batch_dir).glob(f"{prefix}.*.{extension}"))
        )
        self.key = key

    def __iter__(self) -> Iterator[Tensor]:
        for path in self.paths:
            batch = st.load_file(path)[self.key]
            for z in batch:
                yield z


def load_tensor_batched(
    batch_dir: str | Path,
    prefix: str = "batch",
    extension: str = "st",
    key: str = "",
    mask: np.ndarray | Tensor | None = None,
    max_n_batches: int | None = None,
    device: Any = None,
    batch_size: int = 256,
    num_workers: int = 0,
    tqdm_style: Literal["notebook", "console", "none"] | None = None,
) -> Tensor:
    """
    Loads a batched tensor in one go. See `nlnas.datasets.save_tensor_batched`
    and `nlnas.datasets.BatchedTensorDataset` for the precise meaning of the
    arguments.

    Args:
        batch_dir (str | Path):
        prefix (str, optional):
        extension (str, optional):
        key (str, optional):
        mask (np.ndarray | Tensor | None, optional):
        max_n_batches (int | None, optional):
        device (Any, optional): If left to `None`, uses CUDA if it is available,
            otherwise falls back to CPU. Setting `cuda` while CUDA isn't
            available will **silently** fall back to CPU.
        batch_size (int, optional): Defaults to 256.
        num_workers (int, optional): Defaults to 0, meaning single-process data
            loading.
        tqdm_style (Literal['notebook', 'console', 'none'] | None, optional):

    Returns:
        Tensor:
    """
    ds = BatchedTensorDataset(
        batch_dir=batch_dir, prefix=prefix, extension=extension
    )
    dl = EMETDDataLoader(
        ds,
        batch_dir=batch_dir,
        prefix=prefix,
        extension=extension,
        max_n_batches=max_n_batches,
    )
    dl = EMETDDataLoader(
        ds,
        key=key,
        mask=mask,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    everything = make_tqdm(tqdm_style)(dl, "Loading", leave=False)
    return torch.cat(list(everything), dim=0)


def save_tensor_batched(
    x: Tensor | np.ndarray | list,
    output_dir: str | Path,
    prefix: str = "batch",
    extension: str = "st",
    key: str = "",
    batch_size: int = 1024,
    tqdm_style: Literal["notebook", "console", "none"] | None = None,
) -> None:
    """
    Saves a tensor in batches of `batch_size` elements. The files will be named

        output_dir/<prefix>.<batch_idx>.<extension>

    The batches are saved using
    [Safetensors](https://huggingface.co/docs/safetensors/index).

    The `batch_idx` string is 4 digits long, so would be great if you could
    adjust the batch size so that there are less than 10000 batches :]

    Args:
        x (Tensor | np.ndarray | list):
        output_dir (str):
        prefix (str, optional):
        extension (str, optional): Without the first `.`. Defaults to `st`.
        key (str, optional): The key to use when saving the file. Batches are
            stored in safetensor files, which are essentially dictionaries. This
            arg specifies which key contains the data of interest.
        batch_size (int, optional): Defaults to $1024$.
        tqdm_style (Literal["notebook", "console", "none"] | None, optional):
            Progress bar style.
    """
    batches = to_tensor(x).split(batch_size)
    t = make_tqdm(tqdm_style)
    for i, batch in enumerate(t(batches, "Saving", leave=False)):
        data = {key: batch}
        st.save_file(data, Path(output_dir) / f"{prefix}.{i:04}.{extension}")
