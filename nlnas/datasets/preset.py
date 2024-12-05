"""
Some classical dataset configurations that I use often, ready to be instanciated
in one line :)
"""

from typing import Any, Callable

from .huggingface import HuggingFaceDataset

DATASET_PRESETS_CONFIGURATIONS: dict[str, dict[str, Any]] = {
    "cats_vs_dogs": {
        "dataset_name": "microsoft/cats_vs_dogs",
        "train_split": "train[:80%]",
        "val_split": "train[80%:]",
        "test_split": "train",
        "image_key": "image",
        "label_key": "labels",
    },
    "cifar10": {
        "dataset_name": "cifar10",
        "train_split": "train[:80%]",
        "val_split": "train[80%:]",
        "test_split": "test",
        "image_key": "img",
        "label_key": "label",
    },
    "cifar100": {
        "dataset_name": "cifar100",
        "train_split": "train[:80%]",
        "val_split": "train[80%:]",
        "test_split": "test",
        "image_key": "img",
        "label_key": "fine_label",
    },
    "eurosat-rgb": {
        "dataset_name": "timm/eurosat-rgb",
        "train_split": "train",
        "val_split": "validation",
        "test_split": "test",
        "image_key": "image",
        "label_key": "label",
    },
    "fashion_mnist": {
        "dataset_name": "zalando-datasets/fashion_mnist",
        "train_split": "train[:80%]",
        "val_split": "train[80%:]",
        "test_split": "test",
        "image_key": "image",
        "label_key": "label",
    },
    "food101": {
        "dataset_name": "ethz/food101",
        "train_split": "train[:80%]",
        "val_split": "train[80%:]",
        "test_split": "validation",
        "image_key": "image",
        "label_key": "label",
    },
    "imagenet-1k": {
        "dataset_name": "ilsvrc/imagenet-1k",
        "train_split": "train[:80%]",
        "val_split": "train[80%:]",
        "test_split": "validation",
        "image_key": "image",
        "label_key": "label",
    },
    "mnist": {
        "dataset_name": "ylecun/mnist",
        "train_split": "train[:80%]",
        "val_split": "train[80%:]",
        "test_split": "test",
        "image_key": "image",
        "label_key": "label",
    },
    "oxford-iiit-pet": {
        "dataset_name": "timm/oxford-iiit-pet",
        "train_split": "train[:80%]",
        "val_split": "train[80%:]",
        "test_split": "test",
        "image_key": "image",
        "label_key": "label",
    },
    "resisc45": {
        "dataset_name": "timm/resisc45",
        "train_split": "train",
        "val_split": "validation",
        "test_split": "test",
        "image_key": "image",
        "label_key": "label",
    },
}


def get_dataset(
    dataset_name: str,
    image_processor: str | Callable | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    **kwargs: Any,
) -> tuple[HuggingFaceDataset, dict[str, Any]]:
    """
    Returns a `HuggingFaceDataset` instance from the `dataset_name` key in the
    `DATASET_PRESETS_CONFIGURATIONS` dictionary.

    Example:

        ds, _ = get_dataset("cifar10", "microsoft/resnet-18", batch_size=64, num_workers=4)

    Args:
        dataset_name (str): See `DATASET_PRESETS_CONFIGURATIONS`
        image_processor (str | Callable | None): The image processor to use. If
            a str is provided, it is assumed to be a model name, and the image
            processor will be constructed accordingly using
            `nlnas.classifiers.get_classifier_cls` and
            `nlnas.classifiers.BaseClassifier.get_image_processor`.
        batch_size (int | None): If provided, will be added to the dataloader
            parameters for all dataset splits (train, val, test, and predict).
        num_workers (int | None): If provided, will be added to the dataloader
            parameters for all dataset splits (train, val, test, and predict).
        **kwargs: Passed to the `HuggingFaceDataset` constructor

    Returns:
        The `HuggingFaceDataset` instance and the configuration dictionary
    """
    config = DATASET_PRESETS_CONFIGURATIONS[dataset_name]
    if isinstance(image_processor, str):
        from ..classifiers import get_classifier_cls

        cls = get_classifier_cls(image_processor)
        config["image_processor"] = cls.get_image_processor(image_processor)
    else:
        config["image_processor"] = image_processor
    dl_kw = {}
    if batch_size is not None:
        dl_kw["batch_size"] = batch_size
    if num_workers is not None:
        dl_kw["num_workers"] = num_workers
    if dl_kw:
        config["train_dl_kwargs"] = dl_kw
        config["val_dl_kwargs"] = dl_kw.copy()
        config["test_dl_kwargs"] = dl_kw.copy()
        config["predict_dl_kwargs"] = dl_kw.copy()
    config.update(kwargs)
    return HuggingFaceDataset(**config), config
