"""See the `nlnas.datasets.BatchedTensorDataset` class documentation."""

from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import torch
from safetensors import torch as st
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset

from ..utils import make_tqdm, to_tensor


class _ProjectionDataset(IterableDataset):
    """
    Consider a dataset that yields tuples (of anything). This dataset wraps it
    to only select the `i`-th component of said tuple.
    """

    dataset: IterableDataset
    i: int

    def __init__(self, dataset: IterableDataset, i: int) -> None:
        super().__init__()
        self.dataset, self.i = dataset, i

    def __iter__(self) -> Iterator:
        for x in self.dataset:
            yield x[self.i]


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

            batch_dir/<prefix>.<unique_id>.<extension>

        Safetensor files are essentially dictionaries, and each batch file is
        expected to contain:
        * `key` (see below): some `(B, ...)` data tensor.
        * `"_idx"`: a `(B,)` integer tensor.

        When iterating over this dataset, it will yield `(data, idx)` pairs,
        where `data` is a row of the dataset, and `idx` is an int (as a `(,)`
        int tensor).

        Warning:
            The list of batch files will be globed from `batch_dir` upon
            instantiation of this dataset class. It will not be updated
            afterwards.

        Warning:
            Batches are loaded in the order they are found in the filesystem.
            Don't expect this order to be the same as the order in which the
            data has been generated.

        Args:
            batch_dir (str | Path):
            prefix (str, optional):
            extension (str, optional): Without the first `.`. Defaults to `st`.
            key (str, optional): The key to use when saving the file. Batches
                are stored in safetensor files, which are essentially
                dictionaries.  This arg specifies which key contains the data of
                interest. Cannot be `"_idx"`.
        """
        if key == "_idx":
            raise ValueError("Key cannot be '_idx'.")
        self.paths = list(Path(batch_dir).glob(f"{prefix}.*.{extension}"))
        self.key = key

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        for path in self.paths:
            data = st.load_file(path)
            for z, i in zip(data[self.key], data["_idx"]):
                yield z, i

    def extract_idx(self) -> tuple[IterableDataset, Tensor]:
        """
        Splits this dataset in two. The first one yeilds the data, the second
        one yields the indices. Then, the index dataset is unrolled into a
        single index tensor (which therefore has shape `(N,)`, where `N` is the
        shape of the dataset).
        """
        from tqdm import tqdm

        a, b = _ProjectionDataset(self, 0), _ProjectionDataset(self, 1)
        dl = DataLoader(b, batch_size=256, num_workers=1)
        dl = tqdm(dl, "Extracting indices", leave=False)
        # TODO: setting num_workers to > 1 makes the index tensor n_workers
        # times too long... problem with tqdm?
        return a, torch.cat(list(dl), dim=0)

    def load(
        self,
        batch_size: int = 256,
        num_workers: int = 0,
        tqdm_style: Literal["notebook", "console", "none"] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Loads a batched tensor in one go. See `BatchedTensorDataset.save`.

        Args:
            batch_size (int, optional): Defaults to 256. Does not impact the
                actual result.
            num_workers (int, optional): Defaults to 0, meaning single-process
                data loading.
            tqdm_style (Literal['notebook', 'console', 'none'] | None,
                optional):

        Returns:
            A tuple `(data, idx)` where `data` is a `(N, ...)` tensor and `idx`
            is `(N,)` int tensor.
        """
        dl = DataLoader(self, batch_size=batch_size, num_workers=num_workers)
        u = make_tqdm(tqdm_style)(dl, "Loading", leave=False)
        v = list(zip(*u))
        return torch.cat(v[0], dim=0), torch.cat(v[1], dim=0)

    @staticmethod
    def save(
        x: Tensor | np.ndarray | list,
        output_dir: str | Path,
        prefix: str = "batch",
        extension: str = "st",
        key: str = "",
        batch_size: int = 256,
        tqdm_style: Literal["notebook", "console", "none"] | None = None,
    ) -> None:
        """
        Saves a tensor in batches of `batch_size` elements. The files will be
        named

            output_dir/<prefix>.<batch_idx>.<extension>

        The batches are saved using
        [Safetensors](https://huggingface.co/docs/safetensors/index).
        Safetensors files are essentially dictionaries, and each batch file is
        structured as follows:
        * `key` (see below): some `(batch_size, ...)` slice from `x`,
        * `"_idx"`: a `(batch_size,)` integer tensor containing the indices in
          `x`.

        The `batch_idx` string is 4 digits long, so would be great if you could
        adjust the batch size so that there are less than 10000 batches :]

        Args:
            x (Tensor | np.ndarray | list):
            output_dir (str):
            prefix (str, optional):
            extension (str, optional): Without the first `.`. Defaults to `st`.
            key (str, optional): The key to use when saving the file. Batches
                are stored in safetensor files, which are essentially
                dictionaries.  This arg specifies which key contains the data of
                interest. Cannot be `"_idx"`.
            batch_size (int, optional): Defaults to $256$.
            tqdm_style (Literal["notebook", "console", "none"] | None,
                optional): Progress bar style.
        """
        batches = to_tensor(x).split(batch_size)
        t = make_tqdm(tqdm_style)
        for i, batch in enumerate(t(batches, "Saving", leave=False)):
            data = {
                key: batch,
                "_idx": torch.arange(i * batch_size, (i + 1) * batch_size),
            }
            st.save_file(
                data, Path(output_dir) / f"{prefix}.{i:04}.{extension}"
            )
