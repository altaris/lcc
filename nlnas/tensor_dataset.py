"""Defines a pytorch dataset loaded from an numpy NPZ file"""
__docformat__ = "google"

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchvision
from loguru import logger as logging
from safetensors import torch as st
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


class TensorDataset(torch.utils.data.Dataset):
    """
    Defines a pytorch dataset loaded from an numpy NPZ file. The NPZ file is
    expected to have the following structure:
    * `x`: a `(N, ...)` array of data;
    * `y`: a `(N, ...)` array of labels.

    Note that both arrays are held in memory.

    For example,
    * `x`: a `(N, C, W, H)` array of images with `C` channels;
    * `y`: a `(N,)` array of class labels.

    TODO: Change name?
    """

    x: torch.Tensor
    y: torch.Tensor

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __iter__(self):
        return zip(self.x, self.y)

    def __len__(self):
        return self.x.shape[0]

    def __init__(
        self,
        x: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
    ) -> None:
        super().__init__()
        self.x = x if isinstance(x, torch.Tensor) else torch.Tensor(x)
        self.y = y if isinstance(y, torch.Tensor) else torch.Tensor(y)

    def concatenate(self, other: "TensorDataset") -> "TensorDataset":
        """
        Returns a new dataset containing all data and labels from the current
        dataset and `other`
        """
        return TensorDataset(
            torch.concatenate([self.x, other.x]),
            torch.concatenate([self.y, other.y]),
        )

    @staticmethod
    def from_torch_dataset(
        torch_ds: torch.utils.data.IterableDataset,
    ) -> "TensorDataset":
        """Self-explanatory"""
        x, y = [], []
        for a, b in torch_ds:
            x.append(a)
            y.append(b)
        return TensorDataset(torch.stack(x), torch.Tensor(y))

    @staticmethod
    def from_torchvision_dataset(name: str, **kwargs) -> "TensorDataset":
        """
        Self-explanatory, but unlike `from_torch_dataset`, takes a (lower case)
        dataset name. See
        [here](https://pytorch.org/vision/stable/datasets.html#image-classification)
        for the list. You should probably specify transforms in the `kwargs`.
        """
        tvds = torchvision.datasets
        tvdscls = {n.lower(): getattr(tvds, n) for n in tvds.__all__}
        dscls, p = tvdscls[name], Path.home() / "torchvision" / "datasets"
        logging.debug("Loading torchvision dataset '{}' in '{}'", name, p)
        a_ds = dscls(p, download=True, train=True, **kwargs)
        b_ds = dscls(p, download=False, train=False, **kwargs)
        a_tds = TensorDataset.from_torch_dataset(a_ds)
        b_tds = TensorDataset.from_torch_dataset(b_ds)
        return a_tds.concatenate(b_tds)

    @staticmethod
    def load(path: Path) -> "TensorDataset":
        """Self-explanatory"""
        data = st.load_file(path)
        return TensorDataset(data["x"], data["y"])

    def plot(
        self,
        n_images: int = 9,
        n_columns: int = 3,
        imshow_kwargs: dict | None = None,
    ):
        """
        Plots the first n images in a grid. Returns a
        `matplotlib.figure.Figure`.
        """
        # pylint: disable=import-outside-toplevel
        import matplotlib.pyplot as plt

        imshow_kwargs = imshow_kwargs or {}
        idxs = np.array_split(np.arange(n_images, dtype=np.int32), n_columns)
        n_rows = len(idxs)
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_columns, squeeze=False)
        for row_idx, row in enumerate(idxs):
            for col_idx, idx in enumerate(row):
                img = self.x[idx]
                if len(img) != 3:
                    img = img[0]
                ax = axs[row_idx, col_idx]
                ax.imshow(img.numpy(), **imshow_kwargs)
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.tight_layout()
        return fig

    def save(self, path: Path):
        """Self-explanatory"""
        st.save_file({"x": self.x, "y": self.y}, path)

    def shuffle(self) -> None:
        """Shuffles the dataset inplace"""
        p = torch.randperm(len(self.x))
        self.x, self.y = self.x[p], self.y[p]

    def train_test_split(
        self, **kwargs
    ) -> Tuple["TensorDataset", "TensorDataset"]:
        """
        Performs a train/test split using
        `sklearn.model_selection.train_test_split`
        """
        x_train, x_test, y_train, y_test = train_test_split(
            self.x, self.y, **kwargs
        )
        return TensorDataset(x_train, y_train), TensorDataset(x_test, y_test)

    def train_test_split_dl(self, **kwargs) -> Tuple[DataLoader, DataLoader]:
        """
        Same as `tdt.tensor_dataset.TensorDataset.train_test_split`, but
        returns pytorch data loaders instead.
        """
        t, v = self.train_test_split(**kwargs)
        dlkw = {"batch_size": 256, "pin_memory": True}
        return DataLoader(t, **dlkw), DataLoader(v, **dlkw)  # type: ignore

    def y_to_one_hot(self, num_classes: int):
        """Converts the label tensor `self.y` to a one-hot array"""
        self.y = torch.nn.functional.one_hot(
            self.y.to(torch.long), num_classes
        ).to(torch.float32)
