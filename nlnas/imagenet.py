"""The ImageNet dataset as a lightning datamodule"""

from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
import turbo_broccoli as tb
from torch.utils.data import DataLoader, Dataset, IterableDataset, Sampler
from tqdm import tqdm

from .tv_dataset import DEFAULT_DATALOADER_KWARGS, DEFAULT_DOWNLOAD_PATH


def _choice(
    a: torch.Tensor,
    n: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Analogous to
    [`numpy.random.choice`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html)
    except the selection is without replacement the selection distribution is uniform.
    """
    idx = torch.randperm(len(a), generator=generator)
    if n is not None:
        idx = idx[:n]
    return a[idx]


class BalancedBatchSampler(Sampler[list[int]]):
    """
    A batch sampler where the classes are represented (roughly) equally in each
    batch.

        dataset = torchvision.datasets.MNIST(root="/path/to/somewhere")
        dataloader = DataLoader(
            dataset,
            batch_sampler=BalancedBatchSampler(
                dataset, batch_size=2048, n_classes_per_batch=10, seed=0
            ),
        )

    Then sampler can't run in a distributed manner (yet). If using with pytorch
    lightning, don't forget to construct your trainer with
    `use_distributed_sampler=False`.
    """

    class _Iterator(Iterator[list[int]]):
        batch_size: int  # Number of batches to generate left
        classes: torch.Tensor
        generator: torch.Generator
        n_batches: int
        n_classes_per_batch: int
        y: torch.Tensor

        def __init__(
            self,
            y: torch.Tensor,
            batch_size: int,
            n_classes_per_batch: int,
            n_batches: int,
            seed: int | None = None,
        ):
            self.y, self.classes = y, torch.unique(y)
            if len(self.classes) < n_classes_per_batch:
                raise ValueError(
                    "The number of classes per batch cannot exceed the actual "
                    "number of classes"
                )
            self.batch_size, self.n_batches = batch_size, n_batches
            self.n_classes_per_batch = n_classes_per_batch
            self.generator = torch.Generator()
            if seed is not None:
                self.generator.manual_seed(seed)

        def __iter__(self) -> Iterator[list[int]]:
            return self

        def __next__(self) -> list[int]:
            if self.n_batches <= 0:
                raise StopIteration
            self.n_batches -= 1
            classes = _choice(
                self.classes,
                n=self.n_classes_per_batch,
                generator=self.generator,
            )
            r = torch.arange(len(self.y))
            idx = [
                _choice(
                    r[self.y == i],
                    n=self.batch_size // self.n_classes_per_batch,
                    generator=self.generator,
                )
                for i in classes
            ]
            if reminder := self.batch_size % len(self.classes):
                idx += [_choice(r, n=reminder, generator=self.generator)]
            return torch.concat(idx).int().tolist()

    batch_size: int
    n_batches: int
    n_classes_per_batch: int
    seed: int | None
    y: torch.Tensor

    def __init__(
        self,
        y: IterableDataset | torch.Tensor | np.ndarray,
        batch_size: int,
        n_classes_per_batch: int,
        n_batches: int | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        if isinstance(y, np.ndarray):
            self.y = torch.Tensor(y)
        elif isinstance(y, IterableDataset):
            self.y = torch.Tensor([e[-1] for e in y])
        else:
            self.y = y
        self.batch_size = batch_size
        self.n_classes_per_batch = n_classes_per_batch
        self.seed = seed
        self.n_batches = n_batches or (len(self.y) // batch_size)

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Iterator[list[int]]:
        return BalancedBatchSampler._Iterator(
            self.y,
            batch_size=self.batch_size,
            n_classes_per_batch=self.n_classes_per_batch,
            n_batches=self.n_batches,
            seed=self.seed,
        )


class ImageNet(pl.LightningDataModule):
    """
    The ImageNet dataset (from torchvision) wrapped inside a
    [`LightningDataModule`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningDataModule.html)
    """

    download_path: Path
    train_dataset: Dataset | None = None
    transform: Callable
    val_dataset: Dataset | None = None

    def __init__(
        self,
        transform: Callable,
        download_path: str | Path = DEFAULT_DOWNLOAD_PATH,
    ) -> None:
        """
        Args:
            dataloader_kwargs (dict[str, Any] | None, optional):
            download_path (str | Path, optional):
        """
        super().__init__()
        self.transform = transform
        self.download_path = Path(download_path)

    def predict_dataloader(self) -> DataLoader:
        """Not implemented"""
        raise NotImplementedError

    def prepare_data(self) -> None:
        """
        Overrides
        [pl.LightningDataModule.prepare_data](https://lightning.ai/docs/pytorch/stable/data/datamodule.html#prepare-data).
        This is automatically called so don't worry about it.
        """

    def setup(self, stage: str) -> None:
        """
        Overrides
        [pl.LightningDataModule.setup](https://lightning.ai/docs/pytorch/stable/data/datamodule.html#setup).
        This is automatically called so don't worry about it.
        """
        self.train_dataset = torchvision.datasets.ImageNet(
            root=str(self.download_path),
            transform=self.transform,
            split="train",
        )
        self.val_dataset = torchvision.datasets.ImageNet(
            root=str(self.download_path),
            transform=self.transform,
            split="val",
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns the test dataloader. The test dataset must have been loaded
        before calling this method using `TorchvisionDataset.setup('test')`
        """
        if self.val_dataset is None:
            raise RuntimeError(
                "The dataset has not been loaded. Call "
                "`TorchvisionDataset.setup('test')` before using "
                "this datamodule."
            )
        return self.val_dataloader()

    def train_dataloader(self) -> DataLoader:
        """
        Returns the training dataloader. The training dataset must have been
        loaded before calling this method using
        `TorchvisionDataset.setup('fit')`
        """
        # pylint: disable=duplicate-code
        if self.train_dataset is None:
            raise RuntimeError(
                "The dataset has not been loaded. Call "
                "`TorchvisionDataset.setup('fit')` before using "
                "this datamodule."
            )
        kw = DEFAULT_DATALOADER_KWARGS.copy()
        kw["shuffle"] = False
        # return DataLoader(dataset=self.train_dataset, **kw)
        h = tb.GuardedBlockHandler(self.download_path / "train_y.st")
        for _ in h.guard():
            # Should take about 10mins
            dl = DataLoader(self.train_dataset, batch_size=2048, num_workers=8)
            progress = tqdm(
                dl,
                desc="Constructing label vector",
                leave=False,
            )
            y = torch.concat([e[-1] for e in progress]).int()
            h.result = {"y": y}
        return DataLoader(
            dataset=self.train_dataset,
            batch_sampler=BalancedBatchSampler(
                h.result["y"],
                batch_size=2048,
                n_classes_per_batch=10,
                # seed=0,
            ),
            # num_workers=DEFAULT_DATALOADER_KWARGS["num_workers"],
            num_workers=24,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns the validation dataloader. The validation dataset must have
        been loaded before calling this method using
        `TorchvisionDataset.setup('fit')` (yes, `'fit'`, this is not a typo)
        """
        # pylint: disable=duplicate-code
        if self.val_dataset is None:
            raise RuntimeError(
                "The dataset has not been loaded. Call "
                "`TorchvisionDataset.setup('fit')` before using "
                "this datamodule."
            )
        kw = DEFAULT_DATALOADER_KWARGS.copy()
        kw["shuffle"] = False
        return DataLoader(dataset=self.val_dataset, **kw)
