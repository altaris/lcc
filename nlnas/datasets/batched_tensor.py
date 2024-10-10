"""See the `nlnas.datasets.BatchedTensorDataset` class documentation."""

from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import torch
from safetensors import torch as st
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from ..utils import make_tqdm, to_tensor


class BatchedTensorDataset(Dataset):
    """
    Dataset that load tensor batches produced by
    `nlnas.datasets.save_tensor_batched`.
    """

    batch_size: int | None
    device: Any
    key: str
    mask: Tensor | None
    paths: list[Path]
    transform: Callable[[Tensor], Tensor] | None

    def __init__(
        self,
        batch_dir: str | Path,
        prefix: str = "batch",
        extension: str = "st",
        key: str = "",
        mask: np.ndarray | torch.Tensor | None = None,
        batch_size: int | None = None,
        max_n_batches: int | None = None,
        transform: Callable[[Tensor], Tensor] | None = None,
        device: Literal["cpu", "cuda"] | None = None,
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
            mask (np.ndarray | Tensor | None, optional): A boolean mask of shape
                `(N,)`, where $N$ is at least the length of the dataset. If
                provided, only the rows for which the mask is `True` will be
                kept. This means that some batches might be empty.
            batch_size (int | None, optional): If `mask` is provided, then so
                must this argument. Note that the last batch is allowed to be
                smaller than `batch_size`, which happens when the dataset isn't
                evenly divisible in `batch_size` batches.
            max_n_batches (int | None, optional): Limit the dataset to the first
                `max_n_batches` batches.
            transform (Callable[[Tensor], Tensor] | None, optional): An optional
                transform (i.e. function) that will be applied to tensors before
                being returned in `__getitem__`. Note that the tensor will be
                moved to the correct device before being fed to the transform.
            device (Literal['cpu', 'cuda'] | None, optional): Move the batches
                to the given device upon loading.
        """
        if mask is not None and batch_size is None:
            raise ValueError(
                "If a mask is provided, then so must be the batch size."
            )
        self.paths = list(
            sorted(Path(batch_dir).glob(f"{prefix}.*.{extension}"))
        )
        if max_n_batches is not None:
            self.paths = self.paths[:max_n_batches]
        self.key, self.batch_size, self.device = key, batch_size, device
        self.mask = to_tensor(mask) if mask is not None else None
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tensor:
        batch = st.load_file(self.paths[idx])[self.key]
        if self.mask is not None:
            assert self.batch_size is not None
            j = idx * self.batch_size
            m = self.mask[j : j + batch.shape[0]]
            z = batch[m]
        else:
            z = batch
        z = z.to(self.device)
        z = self.transform(z) if self.transform is not None else z
        return z


def load_tensor_batched(
    batch_dir: str | Path,
    prefix: str = "batch",
    extension: str = "st",
    key: str = "",
    mask: np.ndarray | Tensor | None = None,
    batch_size: int | None = None,
    max_n_batches: int | None = None,
    tqdm_style: Literal["notebook", "console", "none"] | None = None,
    device: Literal["cpu", "cuda"] | None = None,
    num_workers: int = 0,
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
        batch_size (int | None, optional):
        max_n_batches (int | None, optional):
        tqdm_style (Literal['notebook', 'console', 'none'] | None, optional):
        device (Literal['cpu', 'cuda'] | None, optional): Defaults to None.
        num_workers (int, optional): Defaults to 0, meaning single-process data
            loading.

    Returns:
        Tensor:
    """
    ds = BatchedTensorDataset(
        batch_dir=batch_dir,
        prefix=prefix,
        extension=extension,
        key=key,
        mask=mask,
        batch_size=batch_size,
        max_n_batches=max_n_batches,
        device=device,
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers)
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
