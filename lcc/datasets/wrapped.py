"""See `lcc.datasets.WrappedDataset` class documentation."""

from typing import Any, Callable, Literal, TypeAlias

import pytorch_lightning as pl
from datasets import Dataset as HuggingFaceDataset
from torch.utils.data import DataLoader, Dataset

from lcc.logging import r0_debug

DEFAULT_DATALOADER_KWARGS: dict[str, Any] = {
    "batch_size": 1024,
    "num_workers": 16,
    "persistent_workers": True,
    "pin_memory": False,
    "shuffle": False,
    "drop_last": False,
}
"""
Default parameters for [pytorch
dataloaders](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader).
"""

DatasetOrDatasetFactory: TypeAlias = (
    Callable[[], Dataset]
    | Dataset
    | Callable[[], HuggingFaceDataset]
    | HuggingFaceDataset
)
"""
Types that are acceptable as dataset or dataloader arguments when
constructing a `WrappedDataset`
"""


class WrappedDataset(pl.LightningDataModule):
    """
    A dataset wrapped inside a
    [`LightningDataModule`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningDataModule.html)
    """

    train: DatasetOrDatasetFactory
    val: DatasetOrDatasetFactory
    test: DatasetOrDatasetFactory | None
    predict: DatasetOrDatasetFactory | None
    train_dl_kwargs: dict[str, Any]
    val_dl_kwargs: dict[str, Any]
    test_dl_kwargs: dict[str, Any]
    predict_dl_kwargs: dict[str, Any]

    _prepared: bool = False

    def __init__(
        self,
        train: DatasetOrDatasetFactory,
        val: DatasetOrDatasetFactory | None = None,
        test: DatasetOrDatasetFactory | None = None,
        predict: DatasetOrDatasetFactory | None = None,
        train_dl_kwargs: dict[str, Any] | None = None,
        val_dl_kwargs: dict[str, Any] | None = None,
        test_dl_kwargs: dict[str, Any] | None = None,
        predict_dl_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            train (DatasetOrDatasetFactory): A dataset or dataloader for
                training. Can be a callable without argument that returns said
                dataset or dataloader. In this case, it will be called during
                the preparation phase (so only on rank 0 in the case of
                multi-device or multi-node training).
            val (DatasetOrDatasetFactory | None, optional): Defaults to
                whatever was passed to the `train` argument
            test (DatasetOrDatasetFactory | None, optional): Defaults to
                `None`
            predict (DatasetOrDatasetFactory | None, optional): Defaults
                to `None`
            train_dl_kwargs (dict[str, Any] | None, optional): If
                `train` is a dataset or callable that return a dataset, this
                dictionary will be passed to the dataloader constructor.
                Defaults to `lcc.datasets.DEFAULT_DATALOADER_KWARGS`.
            val_dl_kwargs (dict[str, Any] | None, optional):
                Analogous to `train_dl_kwargs`, but defaults to (a copy of)
                `train_dl_kwargs` instead of
                `lcc.datasets.DEFAULT_DATALOADER_KWARGS`.
            test_dl_kwargs (dict[str, Any] | None, optional):
                Analogous to `train_dl_kwargs`, but defaults to (a copy of)
                `train_dl_kwargs` instead of
                `lcc.datasets.DEFAULT_DATALOADER_KWARGS`.
            predict_dl_kwargs (dict[str, Any] | None, optional):
                Analogous to `train_dl_kwargs`, but defaults to (a copy of)
                `train_dl_kwargs` instead of
                `lcc.datasets.DEFAULT_DATALOADER_KWARGS`.
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

    def get_dataloader(
        self, split: Literal["train", "val", "test", "predict"]
    ) -> DataLoader:
        """
        Get a dataloader by name. The usual methods `train_dataloader` etc. call
        this. Make sure you called `prepare_data` before calling this.
        """
        if split == "train":
            obj, kw = self.train, self.train_dl_kwargs
        elif split == "val":
            obj, kw = self.val, self.val_dl_kwargs
        elif split == "test":
            if self.test is None:
                raise RuntimeError(
                    "Cannot get test dataloader: no test dataset or dataset "
                    "factory has not been specified"
                )
            obj, kw = self.test, self.test_dl_kwargs
        elif split == "predict":
            if self.predict is None:
                raise RuntimeError(
                    "Cannot get prediction dataloader: no prediction dataset "
                    "or dataset factory has not been specified"
                )
            obj, kw = self.predict, self.predict_dl_kwargs
        else:
            raise ValueError(f"Unknown split: '{split}'")
        obj = obj() if callable(obj) else obj
        return DataLoader(obj, **kw)

    def predict_dataloader(self) -> DataLoader:
        """
        Self-explanatory. Make sure you called `prepare_data` before calling
        this.
        """
        return self.get_dataloader("predict")

    def prepare_data(self) -> None:
        """
        Overrides
        [pl.LightningDataModule.prepare_data](https://lightning.ai/docs/pytorch/stable/data/datamodule.html#prepare-data).
        This is automatically called so don't worry about it.
        """
        if self._prepared:
            return
        if callable(self.train):
            r0_debug("Preparing the training dataset/split")
            self.train = self.train()
        if callable(self.val):
            r0_debug("Preparing the validation dataset/split")
            self.val = self.val()
        if callable(self.test):
            r0_debug("Preparing the testing dataset/split")
            self.test = self.test()
        if callable(self.predict):
            r0_debug("Preparing the prediction dataset/split")
            self.predict = self.predict()
        self._prepared = True

    def test_dataloader(self) -> DataLoader:
        """
        Self-explanatory. Make sure you called `prepare_data` before calling
        this.
        """
        return self.get_dataloader("test")

    def train_dataloader(self) -> DataLoader:
        """
        Self-explanatory. Make sure you called `prepare_data` before calling
        this.
        """
        return self.get_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        """
        Self-explanatory. Make sure you called `prepare_data` before calling
        this.
        """
        return self.get_dataloader("val")
