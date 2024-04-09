"""
A dataset wrapped inside a
[`LightningDataModule`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningDataModule.html)
"""

from typing import Any, Callable, Literal, TypeAlias

import pytorch_lightning as pl
import torch
from datasets import Dataset as HuggingFaceDataset
from loguru import logger as logging
from torch.utils.data import DataLoader, Dataset

DEFAULT_DATALOADER_KWARGS: dict[str, Any] = {
    "batch_size": 128 if torch.cuda.is_available() else 32,
    "num_workers": 16,
    "persistent_workers": True,
    "pin_memory": False,
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
    train_dl_kwargs: dict[str, Any]
    val_dl_kwargs: dict[str, Any]
    test_dl_kwargs: dict[str, Any]
    predict_dl_kwargs: dict[str, Any]

    train_dl: DataLoader | None = None
    val_dl: DataLoader | None = None
    test_dl: DataLoader | None = None
    predict_dl: DataLoader | None = None

    _prepared: bool = False

    def __init__(
        self,
        train: DatasetOrDataLoaderOrFactory,
        val: DatasetOrDataLoaderOrFactory | None = None,
        test: DatasetOrDataLoaderOrFactory | None = None,
        predict: DatasetOrDataLoaderOrFactory | None = None,
        train_dl_kwargs: dict[str, Any] | None = None,
        val_dl_kwargs: dict[str, Any] | None = None,
        test_dl_kwargs: dict[str, Any] | None = None,
        predict_dl_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            train (DatasetOrDataLoaderOrFactory): A dataset or dataloader for
                training. Can be a callable without argument that returns said
                dataset or dataloader. In this case, it will be called during
                the preparation phase (so only on rank 0).
            val (DatasetOrDataLoaderOrFactory | None, optional): Defaults to
                whatever was passed to the `train` argument
            test (DatasetOrDataLoaderOrFactory | None, optional): Defaults to
                `None`
            predict (DatasetOrDataLoaderOrFactory | None, optional): Defaults
                to `None`
            train_dl_kwargs (dict[str, Any] | None, optional): If
                `train`is a dataset or callable that return datasets, this
                dictionary will be passed to the dataloader constructor.
                Defaults to `DEFAULT_DATALOADER_KWARGS`
            val_dl_kwargs (dict[str, Any] | None, optional):
                Analogous to `train_dl_kwargs`, but defaults to (a copy of)
                `train_dl_kwargs` instead of `DEFAULT_DATALOADER_KWARGS`
            test_dl_kwargs (dict[str, Any] | None, optional):
                Analogous to `train_dl_kwargs`, but defaults to (a copy of)
                `train_dl_kwargs` instead of `DEFAULT_DATALOADER_KWARGS`
            predict_dl_kwargs (dict[str, Any] | None, optional):
                Analogous to `train_dl_kwargs`, but defaults to (a copy of)
                `train_dl_kwargs` instead of `DEFAULT_DATALOADER_KWARGS`
        """
        super().__init__()
        self.train, self.val = train, val if val is not None else train
        self.test, self.predict = test, predict
        self.train_dl_kwargs = (
            train_dl_kwargs or DEFAULT_DATALOADER_KWARGS.copy()
        )
        self.val_dl_kwargs = val_dl_kwargs or self.train_dl_kwargs.copy()
        self.test_dl_kwargs = test_dl_kwargs or self.train_dl_kwargs.copy()
        self.predict_dl_kwargs = (
            predict_dl_kwargs or self.train_dl_kwargs.copy()
        )

    def _get_dl(
        self, split: Literal["train", "val", "test", "predict"]
    ) -> DataLoader:
        if split == "train":
            obj, kw = self.train, self.train_dl_kwargs
        elif split == "val":
            obj, kw = self.val, self.val_dl_kwargs
        elif split == "test":
            obj, kw = self.test, self.test_dl_kwargs
        elif split == "predict":
            obj, kw = self.predict, self.predict_dl_kwargs
        else:
            raise ValueError(f"Unknown split: '{split}'")
        obj = obj() if callable(obj) else obj
        return obj if isinstance(obj, DataLoader) else DataLoader(obj, **kw)

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
        if self._prepared:
            return
        if callable(self.train):
            logging.debug("Preparing the training dataset/split")
            self.train()
        if callable(self.val):
            logging.debug("Preparing the validation dataset/split")
            self.val()
        if callable(self.test):
            logging.debug("Preparing the testing dataset/split")
            self.test()
        if callable(self.predict):
            logging.debug("Preparing the prediction dataset/split")
            self.predict()
        self._prepared = True

    def setup(self, stage: str) -> None:
        """
        Overrides
        [pl.LightningDataModule.setup](https://lightning.ai/docs/pytorch/stable/data/datamodule.html#setup).
        This is automatically called so don't worry about it.
        """
        if stage == "fit":
            self.train_dl = self._get_dl("train")
            self.val_dl = self._get_dl("val")
        elif stage == "validate":
            self.val_dl = self._get_dl("val")
        elif stage == "test":
            if self.test is None:
                raise RuntimeError(
                    "Cannot setup datamodule in 'test' mode: no test dataset "
                    "was provided"
                )
            self.test_dl = self._get_dl("test")
        elif stage == "predict":
            if self.predict is None:
                raise RuntimeError(
                    "Cannot setup datamodule in 'predict' mode: no predict "
                    "dataset was provided"
                )
            self.predict_dl = self._get_dl("predict")
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
