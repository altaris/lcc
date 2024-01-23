"""Custom torchvision transforms"""

from typing import Callable

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


def dataset_normalization(dataset_name: str) -> Callable:
    """
    Returns a [normalization
    transform](https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html)
    with parameters tailored for a given dataset. The parameters were obtained
    with a snippet that looks like this:

        from torchvision.datasets import *
        from torch.utils.data import DataLoader
        from torchvision import transforms

        ds = Flowers102(
            root="/home/cedric/torchvision/datasets/",
            transform=transforms.Compose(
                [
                    transforms.Resize([512, 512]),  # if no standard size
                    transforms.ToTensor(),
                ]
            ),
            split="train",  # or something like that
            download=True,
        )
        dl = DataLoader(ds, batch_size=256)
        x = torch.concat([a for a, _ in tqdm(dl)])
        mean = [float(x[:,i].mean()) for i in range(x.shape[1])]
        std = [float(x[:,i].std()) for i in range(x.shape[1])]
        print(mean)
        print(std)

    The parameters for `imagenet` where shamelessly stolen from
    [Lightning-Universe/lightning-bolts](https://github.com/Lightning-Universe/lightning-bolts/blob/master/src/pl_bolts/transforms/dataset_normalizations.py).
    """
    parameters = {
        "cifar10": ([0.491, 0.482, 0.446], [0.247, 0.243, 0.261]),
        "cifar100": ([0.491, 0.482, 0.446], [0.247, 0.243, 0.261]),
        "fashionmnist": ([0.286], [0.353]),
        "flowers102": ([0.432, 0.381, 0.296], [0.294, 0.246, 0.273]),
        "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "mnist": ([0.130], [0.308]),
        "pcam": ([0.700, 0.538, 0.691], [0.234, 0.277, 0.212]),
        "stl10": ([0.446, 0.439, 0.406], [0.260, 0.256, 0.271]),
    }
    if dataset_name not in parameters:
        raise ValueError(
            "Could not find normalization parameters for dataset "
            f"'{dataset_name}'"
        )
    mean, std = parameters[dataset_name]
    return transforms.Normalize(mean=mean, std=std)
