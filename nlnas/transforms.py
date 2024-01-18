"""Custom torchvision transforms"""

from typing import Callable, Literal

from torch import Tensor
from torchvision import transforms


class EnsureRGB:
    """
    Makes sures that the images (in the form of tensors) have 3 channels:

    * if the input tensor has shape `(3, H, W)`, then nothing is done;
    * if the input tensor has shape `(1, H, W)`, then it is repeated along
      axis 0;
    * otherwise, an `ValueError` is raised.
    """

    def __call__(self, x: Tensor) -> Tensor:
        nc = x.shape[0]
        if not (x.ndim == 3 and nc in [1, 3]):
            raise ValueError(f"Unsupported shape {list(x.shape)}")
        if nc == 1:
            return x.repeat(3, 1, 1)
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def cifar10_normalization() -> Callable:
    """
    Normalization transform for the cifar10 dataset. Inspired from
    [Lightning-Universe/lightning-bolts](https://github.com/Lightning-Universe/lightning-bolts/blob/master/src/pl_bolts/transforms/dataset_normalizations.py)
    but the constants were slightly adjusted:

        from nlnas.tv_dataset import TorchvisionDataset

        ds = TorchvisionDataset("cifar10")
        ds.setup("fit")
        ds.setup("test")
        x = torch.concat(
            [a for a, _ in ds.train_dataloader()]
            + [a for a, _ in ds.test_dataloader()]
        )
        mean = [float(x[:,i].mean()) for i in range(x.shape[1])]
        std = [float(x[:,i].std()) for i in range(x.shape[1])]

        print(mean, std)
    """
    return transforms.Normalize(
        mean=[0.491, 0.482, 0.446], std=[0.247, 0.243, 0.261]
    )


def fashionmnist_normalization() -> Callable:
    """
    Normalization transform for the mnist dataset. Inspired from
    [Lightning-Universe/lightning-bolts](https://github.com/Lightning-Universe/lightning-bolts/blob/master/src/pl_bolts/transforms/dataset_normalizations.py).
    but the constants were slightly adjusted:

        from nlnas.tv_dataset import TorchvisionDataset

        ds = TorchvisionDataset("fashionmnist")
        ds.setup("fit")
        ds.setup("test")
        x = torch.concat(
            [a for a, _ in ds.train_dataloader()]
            + [a for a, _ in ds.test_dataloader()]
        )
        mean = float(x.mean())
        std = float(x.std())

        print(mean, std)
    """
    return transforms.Normalize(mean=[0.286], std=[0.353])


def imagenet_normalization() -> Callable:
    """
    Normalization transform for the imagenet dataset. Inspired from
    [Lightning-Universe/lightning-bolts](https://github.com/Lightning-Universe/lightning-bolts/blob/master/src/pl_bolts/transforms/dataset_normalizations.py).
    """
    return transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )


def mnist_normalization() -> Callable:
    """
    Normalization transform for the mnist dataset. Shamelessly stolen from
    [Lightning-Universe/lightning-bolts](https://github.com/Lightning-Universe/lightning-bolts/blob/master/src/pl_bolts/transforms/dataset_normalizations.py).
    but the constants were slightly adjusted:

        from nlnas.tv_dataset import TorchvisionDataset

        ds = TorchvisionDataset("mnist")
        ds.setup("fit")
        ds.setup("test")
        x = torch.concat(
            [a for a, _ in ds.train_dataloader()]
            + [a for a, _ in ds.test_dataloader()]
        )
        mean = float(x.mean())
        std = float(x.std())

        print(mean, std)
    """
    return transforms.Normalize(mean=[0.130], std=[0.308])
