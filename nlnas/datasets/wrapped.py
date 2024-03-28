"""
A dataset wrapped inside a
[`LightningDataModule`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningDataModule.html)
"""

from typing import Any, Callable

import pytorch_lightning as pl
import torch
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

DatasetOrDataLoaderOrFactory = (
    Callable[[], Dataset] | Dataset | Callable[[], DataLoader] | DataLoader
)
"""
Types that are acceptable as dataset or dataloader arguments when
constructing a `WrappedDataset`
"""


class WrappedDataset(pl.LightningDataModule):
    """See module documentation"""

    fit: DatasetOrDataLoaderOrFactory
    val: DatasetOrDataLoaderOrFactory
    test: DatasetOrDataLoaderOrFactory | None
    predict: DatasetOrDataLoaderOrFactory | None
    dataloader_kwargs: dict[str, Any]

    fit_dl: DataLoader | None = None
    val_dl: DataLoader | None = None
    test_dl: DataLoader | None = None
    predict_dl: DataLoader | None = None

    def __init__(
        self,
        fit: DatasetOrDataLoaderOrFactory,
        val: DatasetOrDataLoaderOrFactory | None = None,
        test: DatasetOrDataLoaderOrFactory | None = None,
        predict: DatasetOrDataLoaderOrFactory | None = None,
        dataloader_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            fit (DatasetOrDataLoaderOrFactory): A dataset or dataloader for
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
                constructor.
        """
        super().__init__()
        self.fit, self.val = fit, val if val is not None else fit
        self.test, self.predict = test, predict
        self.dataloader_kwargs = dataloader_kwargs or {}

    def _get_dl(self, dataset: DatasetOrDataLoaderOrFactory) -> DataLoader:
        x: Dataset | DataLoader = dataset() if callable(dataset) else dataset
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
        if callable(self.fit):
            self.fit()
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
            self.fit_dl = self._get_dl(self.fit)
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

    def fit_dataloader(self) -> DataLoader:
        """
        Returns the training dataloader. The training dataset must have been
        loaded before calling this method using `setup('fit')`
        """
        if self.fit_dl is None:
            raise RuntimeError(
                "The training dataset has not been loaded. Call "
                "`setup('fit')` before calling this method"
            )
        return self.fit_dl

    def val_dataloader(self) -> DataLoader:
        """
        Returns the valudation dataloader. The valudation dataset must have
        been loaded before calling this method using `setup('fit')` (yes, the
        'fit' stage, this is not a typo)
        """
        if self.val_dl is None:
            raise RuntimeError(
                "The valudation dataset has not been loaded. Call "
                "`setup('fit')` before calling this method (yes, 'fit', this "
                "is not a typo)"
            )
        return self.val_dl
