"""
A dataset wrapped inside a
[`LightningDataModule`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningDataModule.html)
"""

from typing import Any, Callable, TypeAlias

import pytorch_lightning as pl
import torch
from datasets import Dataset as HuggingFaceDataset
from torch.utils.data import DataLoader, Dataset

DEFAULT_DATALOADER_KWARGS: dict[str, Any] = {
    "batch_size": 256 if torch.cuda.is_available() else 64,
    "num_workers": 8,
    "persistent_workers": True,
    "pin_memory": True,
}
"""
Default parameters for [pytorch
dataloaders](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader).
"""

DatasetOrDataLoaderOrFactory: TypeAlias = (
    Callable[[], Dataset]
    | Dataset
    | Callable[[], HuggingFaceDataset]
    | HuggingFaceDataset
    | Callable[[], DataLoader]
    | DataLoader
)
"""
Types that are acceptable as dataset or dataloader arguments when
constructing a `WrappedDataset`
"""


class WrappedDataset(pl.LightningDataModule):
    """See module documentation"""

    train: DatasetOrDataLoaderOrFactory
    val: DatasetOrDataLoaderOrFactory
    test: DatasetOrDataLoaderOrFactory | None
    predict: DatasetOrDataLoaderOrFactory | None
    dataloader_kwargs: dict[str, Any]

    train_dl: DataLoader | None = None
    val_dl: DataLoader | None = None
    test_dl: DataLoader | None = None
    predict_dl: DataLoader | None = None

    def __init__(
        self,
        train: DatasetOrDataLoaderOrFactory,
        val: DatasetOrDataLoaderOrFactory | None = None,
        test: DatasetOrDataLoaderOrFactory | None = None,
        predict: DatasetOrDataLoaderOrFactory | None = None,
        dataloader_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            train (DatasetOrDataLoaderOrFactory): A dataset or dataloader for
                training. Can be a callable without argument that returns said
                dataset or dataloader. In this case, it will be called during
                the preparation phase (so only on rank 0).
            val (DatasetOrDataLoaderOrFactory | None, optional): Defaults to
                whatever was passed to the `fit` argument
            test (DatasetOrDataLoaderOrFactory | None, optional): Defaults to
                `None`
            predict (DatasetOrDataLoaderOrFactory | None, optional): Defaults
                to `None`
            dataloader_kwargs (dict[str, Any] | None, optional): If
                some arguments where datasets (or callable that return
                datasets), this dictionary will be passed to the dataloader
                constructor. Defaults to `DEFAULT_DATALOADER_KWARGS`
        """
        super().__init__()
        self.train, self.val = train, val if val is not None else train
        self.test, self.predict = test, predict
        self.dataloader_kwargs = (
            dataloader_kwargs or DEFAULT_DATALOADER_KWARGS.copy()
        )

    def _get_dl(self, dataset: DatasetOrDataLoaderOrFactory) -> DataLoader:
        x: Dataset | HuggingFaceDataset | DataLoader = (
            dataset() if callable(dataset) else dataset
        )
        return (
            x
            if isinstance(x, DataLoader)
            else DataLoader(x, **self.dataloader_kwargs)
        )

    def predict_dataloader(self) -> DataLoader:
        """
        Returns the prediction dataloader. The prediction dataset must have
        been loaded before calling this method using `setup('predict')`
        """
        if self.predict_dl is None:
            raise RuntimeError(
                "The prediction dataset has not been loaded. Call "
                "`setup('predict')` before calling this method"
            )
        return self.predict_dl

    def prepare_data(self) -> None:
        """
        Overrides
        [pl.LightningDataModule.prepare_data](https://lightning.ai/docs/pytorch/stable/data/datamodule.html#prepare-data).
        This is automatically called so don't worry about it.
        """
        if callable(self.train):
            self.train()
        if callable(self.val):
            self.val()
        if callable(self.test):
            self.test()
        if callable(self.predict):
            self.predict()

    def setup(self, stage: str) -> None:
        """
        Overrides
        [pl.LightningDataModule.setup](https://lightning.ai/docs/pytorch/stable/data/datamodule.html#setup).
        This is automatically called so don't worry about it.
        """
        if stage == "fit":
            self.train_dl = self._get_dl(self.train)
            self.val_dl = self._get_dl(self.val)
        elif stage == "validate":
            self.val_dl = self._get_dl(self.val)
        elif stage == "test":
            if self.test is None:
                raise RuntimeError(
                    "Cannot setup datamodule in 'test' mode: no test dataset "
                    "was provided"
                )
            self.test_dl = self._get_dl(self.test)
        elif stage == "predict":
            if self.predict is None:
                raise RuntimeError(
                    "Cannot setup datamodule in 'predict' mode: no predict "
                    "dataset was provided"
                )
            self.predict_dl = self._get_dl(self.predict)
        else:
            raise ValueError(f"Unknown stage: '{stage}'")

    def test_dataloader(self) -> DataLoader:
        """
        Returns the test dataloader. The test dataset must have been loaded
        before calling this method using `setup('test')`
        """
        if self.test_dl is None:
            raise RuntimeError(
                "The test dataset has not been loaded. Call "
                "`setup('test')` before calling this method"
            )
        return self.test_dl

    def train_dataloader(self) -> DataLoader:
        """
        Returns the training dataloader. The training dataset must have been
        loaded before calling this method using `setup('fit')`
        """
        if self.train_dl is None:
            raise RuntimeError(
                "The training dataset has not been loaded. Call "
                "`setup('fit')` before calling this method"
            )
        return self.train_dl

    def val_dataloader(self) -> DataLoader:
        """
        Returns the valudation dataloader. The valudation dataset must have
        been loaded before calling this method using `setup('fit')` or
        `setup('validate')`
        """
        if self.val_dl is None:
            raise RuntimeError(
                "The valudation dataset has not been loaded. Call "
                "`setup('fit')` or `setup('validate')` before calling this "
                "method"
            )
        return self.val_dl
