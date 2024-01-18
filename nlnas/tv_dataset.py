"""A torchvision dataset wrapped inside a `LightningDataModule`"""

import functools
from pathlib import Path
from typing import Any, Callable

import pytorch_lightning as pl
import torch
import torchvision

from torch.utils.data import DataLoader, Dataset, random_split

from .transforms import EnsureRGB
from .utils import dl_targets


DEFAULT_DATALOADER_KWARGS = {
    "batch_size": 256 if torch.cuda.is_available() else 64,
    "num_workers": 8,
    "persistent_workers": True,
    "pin_memory": True,
    "shuffle": True,
}
"""
Default parameters for [pytorch
dataloaders](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader).
Note that for validation, test, and prediction dataloaders, shuffling is always disabled.
"""

DEFAULT_DOWNLOAD_PATH: Path = Path.home() / "torchvision" / "datasets"
"""
Default download path for torchvision dataset. This constant is actually
defined at runtime by calling
[`Path.home()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.home)
so the value as displayed in the docs may not be accurate.
"""

ALL_DATASETS = {
    "caltech101": torchvision.datasets.Caltech101,
    "caltech256": torchvision.datasets.Caltech256,
    "celeba": torchvision.datasets.CelebA,
    "cifar10": torchvision.datasets.CIFAR10,
    "cifar100": torchvision.datasets.CIFAR100,
    "country211": torchvision.datasets.Country211,
    "dtd": torchvision.datasets.DTD,
    "eurosat": torchvision.datasets.EuroSAT,
    "fashionmnist": torchvision.datasets.FashionMNIST,
    "fer2013": torchvision.datasets.FER2013,
    "fgvcaircraft": torchvision.datasets.FGVCAircraft,
    "flowers102": torchvision.datasets.Flowers102,
    "food101": torchvision.datasets.Food101,
    "gtsrb": torchvision.datasets.GTSRB,
    "inaturalist": torchvision.datasets.INaturalist,
    "imagenet": torchvision.datasets.ImageNet,
    "kmnist": torchvision.datasets.KMNIST,
    "lfwpeople": torchvision.datasets.LFWPeople,
    "lsun": torchvision.datasets.LSUN,
    "mnist": torchvision.datasets.MNIST,
    "omniglot": torchvision.datasets.Omniglot,
    "oxfordiiitpet": torchvision.datasets.OxfordIIITPet,
    "pcam": torchvision.datasets.PCAM,
    "qmnist": torchvision.datasets.QMNIST,
    "renderedsst2": torchvision.datasets.RenderedSST2,
    "semeion": torchvision.datasets.SEMEION,
    "sbu": torchvision.datasets.SBU,
    "stanfordcars": torchvision.datasets.StanfordCars,
    "stl10": torchvision.datasets.STL10,
    "sun397": torchvision.datasets.SUN397,
    "svhn": torchvision.datasets.SVHN,
    "usps": torchvision.datasets.USPS,
}


class TorchvisionDataset(pl.LightningDataModule):
    """
    A torchvision dataset wrapped inside a [`LightningDataModule`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningDataModule.html)

    See also:
        https://pytorch.org/vision/stable/datasets.html#image-classification
    """

    dataloader_kwargs: dict[str, Any]
    dataset_name: str
    download_path: str | Path
    fit_kwargs: dict[str, Any]
    predict_kwargs: dict[str, Any]
    test_kwargs: dict[str, Any]
    train_val_split: float
    transform: Callable

    dataset: Dataset | None = None
    train_dataset: Dataset | None = None
    val_dataset: Dataset | None = None

    def __init__(
        self,
        dataset_name: str,
        transform: Callable | None = None,
        fit_kwargs: dict[str, Any] | None = None,
        test_kwargs: dict[str, Any] | None = None,
        predict_kwargs: dict[str, Any] | None = None,
        dataloader_kwargs: dict[str, Any] | None = None,
        download_path: str | Path = DEFAULT_DOWNLOAD_PATH,
        train_val_split: float = 0.8,
    ) -> None:
        """
        Args:
            dataset_name (str):
            transform (Callable | None, optional):
            fit_kwargs (dict[str, Any] | None, optional): Dataset constructor
                kwargs when this datamodule is setup in `fit` mode
            test_kwargs (dict[str, Any] | None, optional): Dataset constructor
                kwargs when this datamodule is setup in `test` mode
            predict_kwargs (dict[str, Any] | None, optional): Dataset
                constructor kwargs when this datamodule is setup in `predict`
                mode
            dataloader_kwargs (dict[str, Any] | None, optional):
            download_path (str | Path, optional):
            train_val_split (float, optional): If this datamodule is setup in
                `fit` mode, fraction of the dataset to use for training (which
                of course implies that the fraction for validation is `1 -
                train_val_split`). Must be in $(0, 1)$.
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.transform = transform or torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                EnsureRGB(),
            ]
        )
        self.download_path = download_path
        self.fit_kwargs = fit_kwargs or {}
        self.test_kwargs = test_kwargs or {}
        self.predict_kwargs = predict_kwargs or {}
        self.dataloader_kwargs = dataloader_kwargs or DEFAULT_DATALOADER_KWARGS
        self.train_val_split = train_val_split

    @functools.cached_property
    def image_shape(self) -> list[int]:
        """
        Returns the shape of the images in the dataset. This sets up the
        current datamodule in `fit` mode and looks at the first image in the
        validation dataset.
        """
        self.setup("fit")
        return list(next(iter(self.val_dataloader()))[0].shape)[1:]

    @functools.cached_property
    def n_classes(self) -> int:
        """
        Returns the number of classes in the dataset. This sets up the current
        datamodule in `fit` mode and iterates over the whole validation
        dataset.
        """
        self.setup("fit")
        return len(dl_targets(self.val_dataloader()))

    def predict_dataloader(self) -> DataLoader:
        """
        Returns the prediction dataloader. The prediction dataset must have
        been loaded before calling this method using
        `TorchvisionDataset.setup('predict')`
        """
        if self.dataset is None:
            raise RuntimeError(
                "The dataset has not been loaded. Call "
                "`TorchvisionDataset.setup('predict')` before using "
                "this datamodule."
            )
        kw = self.dataloader_kwargs.copy()
        kw["shuffle"] = False
        return DataLoader(dataset=self.dataset, **kw)

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
        if stage not in ["fit", "test", "predict"]:
            raise ValueError(f"Unsupported stage: '{stage}'")
        kwargs = {
            "fit": self.fit_kwargs,
            "test": self.test_kwargs,
            "predict": self.predict_kwargs,
        }[stage]
        factory = ALL_DATASETS[self.dataset_name]
        self.dataset = factory(
            root=str(self.download_path),
            # download=True,  # Necessary first time but then clutters the logs
            transform=self.transform,
            **kwargs,
        )
        assert self.dataset is not None  # For typechecking
        if stage == "fit":
            self.train_dataset, self.val_dataset = random_split(
                self.dataset, [self.train_val_split, 1 - self.train_val_split]
            )

    def test_dataloader(self) -> DataLoader:
        """
        Returns the test dataloader. The test dataset must have been loaded
        before calling this method using `TorchvisionDataset.setup('test')`
        """
        if self.dataset is None:
            raise RuntimeError(
                "The dataset has not been loaded. Call "
                "`TorchvisionDataset.setup('test')` before using "
                "this datamodule."
            )
        kw = self.dataloader_kwargs.copy()
        kw["shuffle"] = False
        return DataLoader(dataset=self.dataset, **kw)

    def train_dataloader(self) -> DataLoader:
        """
        Returns the training dataloader. The training dataset must have been
        loaded before calling this method using
        `TorchvisionDataset.setup('fit')`
        """
        if self.train_dataset is None:
            raise RuntimeError(
                "The dataset has not been loaded. Call "
                "`TorchvisionDataset.setup('fit')` before using "
                "this datamodule."
            )
        return DataLoader(dataset=self.train_dataset, **self.dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        """
        Returns the validation dataloader. The validation dataset must have
        been loaded before calling this method using
        `TorchvisionDataset.setup('fit')` (yes, `'fit'`, this is not a typo)
        """
        if self.val_dataset is None:
            raise RuntimeError(
                "The dataset has not been loaded. Call "
                "`TorchvisionDataset.setup('fit')` before using "
                "this datamodule."
            )
        kw = self.dataloader_kwargs.copy()
        kw["shuffle"] = False
        return DataLoader(dataset=self.val_dataset, **kw)
