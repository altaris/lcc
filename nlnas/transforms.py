"""Custom torchvision transforms"""

from typing import Callable, Literal

from torch import Tensor
from torchvision import transforms


class EnsuresRGB:
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
    Normalization transform for the cifar10 dataset. Shamelessly stolen from
    [Lightning-Universe/lightning-bolts](https://github.com/Lightning-Universe/lightning-bolts/blob/master/src/pl_bolts/transforms/dataset_normalizations.py).
    """
    return transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )


def emnist_normalization(
    split: Literal[
        "balanced", "byclass", "bymerge", "digits", "letters", "mnist"
    ]
) -> Callable:
    """
    Normalization transform for the emnist dataset. Shamelessly stolen from
    [Lightning-Universe/lightning-bolts](https://github.com/Lightning-Universe/lightning-bolts/blob/master/src/pl_bolts/transforms/dataset_normalizations.py).

    Args:
        split (str): Either `balanced`, `byclass`, `bymerge`, `digits`,
        `letters`, or `mnist`.

    See also:
        [`torchvision.datasets.EMNIST`](https://pytorch.org/vision/stable/generated/torchvision.datasets.EMNIST.html#torchvision.datasets.EMNIST)
    """
    stats = {
        "balanced": (0.175, 0.333),
        "byclass": (0.174, 0.332),
        "bymerge": (0.174, 0.332),
        "digits": (0.173, 0.332),
        "letters": (0.172, 0.331),
        "mnist": (0.173, 0.332),
    }

    return transforms.Normalize(mean=stats[split][0], std=stats[split][1])


def imagenet_normalization() -> Callable:
    """
    Normalization transform for the imagenet dataset. Shamelessly stolen from
    [Lightning-Universe/lightning-bolts](https://github.com/Lightning-Universe/lightning-bolts/blob/master/src/pl_bolts/transforms/dataset_normalizations.py).
    """
    return transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
