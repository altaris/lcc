from torchvision import transforms

from nlnas.transforms import EnsureRGB, dataset_normalization

DATASETS = {
    "mnist": transforms.Compose(
        [
            transforms.ToTensor(),
            EnsureRGB(),
            dataset_normalization("mnist"),
            transforms.Resize([64, 64], antialias=True),
        ]
    ),
    "fashionmnist": transforms.Compose(
        [
            transforms.ToTensor(),
            EnsureRGB(),
            dataset_normalization("fashionmnist"),
            transforms.Resize([64, 64], antialias=True),
        ]
    ),
    "cifar10": transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            dataset_normalization("cifar10"),
            transforms.Resize([64, 64], antialias=True),
        ]
    ),
    "cifar100": transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            dataset_normalization("cifar10"),
            transforms.Resize([64, 64], antialias=True),
        ]
    ),
    "stl10": transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            dataset_normalization("stl10"),
        ]
    ),
    "pcam": transforms.Compose(
        [
            transforms.ToTensor(),
            dataset_normalization("pcam"),
        ]
    ),
    "eurosat": transforms.Compose(
        [
            transforms.ToTensor(),
            dataset_normalization("eurosat"),
        ]
    ),
    "semeion": transforms.Compose(
        [
            transforms.ToTensor(),
            dataset_normalization("semeion"),
        ]
    ),
    # "flowers102": transforms.Compose(
    #     [
    #         transforms.Resize([128, 128], antialias=True),
    #         transforms.ToTensor(),
    #         dataset_normalization("flowers102"),
    #     ]
    # ),
}

WEIGHT_EXPONENTS = [0, 1, 3, 5, 10]
"""The weight will actually be $10^{-e}$"""

BATCH_SIZES = [2048]

KS = [5, 25, 50]
