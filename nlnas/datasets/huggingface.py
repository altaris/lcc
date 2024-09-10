"""
A Hugging Face image classification dataset wrapped inside a `WrappedDataset`,
which is itself a
[`LightningDataModule`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningDataModule.html).

Hugging Face image datasets are dict datasets where the image is a PIL image
object. Here, images are converted to tensors using the `image_processor` (if
provided), which brings this closer to the torchvision API. In this case, load
and call Hugging Face models directly. If you do not provide an
`image_processor`, then it is recommended that you use a Hugging Face pipeline
instead.

Since Hugging Face datasets are dict datasets, batches are dicts of tensors
(see the Hugging Face dataset hub for the list of keys). `HuggingFaceDataset`
adds an extra key `_idx` that has the index of the samples in the dataset.

See also:
    https://huggingface.co/datasets?task_categories=task_categories:image-classification
"""

from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
from datasets import Dataset, load_dataset
from torch import Tensor

from .wrapped import WrappedDataset

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "datasets"
"""
Default download path for huggingface dataset.

See also:
    https://huggingface.co/docs/datasets/cache
"""


class HuggingFaceDataset(WrappedDataset):
    """See module documentation"""

    label_key: str

    _datasets: dict[str, Dataset] = {}  # Cache

    def __init__(
        self,
        dataset_name: str,
        fit_split: str = "training",
        val_split: str = "validation",
        test_split: str | None = None,
        predict_split: str | None = None,
        image_processor: Callable | None = None,
        train_dl_kwargs: dict[str, Any] | None = None,
        val_dl_kwargs: dict[str, Any] | None = None,
        test_dl_kwargs: dict[str, Any] | None = None,
        predict_dl_kwargs: dict[str, Any] | None = None,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        classes: list | Tensor | np.ndarray | None = None,
        label_key: str = "label",
    ) -> None:
        """
        Args:
            dataset_name (str): Name of the Hugging Face image classification
                dataset, see
                https://huggingface.co/datasets?task_categories=task_categories:image-classification
            fit_split (str, optional): Name of the split containing the
                training data. See also
                https://huggingface.co/docs/datasets/en/loading#slice-splits
            val_split (str, optional): Analogous to `fit_split`
            test_split (str | None, optional): Analogous to `fit_split`. If
                left to `None`, setting up this datamodule at the `test` stage
                will raise a `RuntimeError`
            predict_split (str | None, optional): Analogous to `fit_split`. If
                left to `None`, setting up this datamodule at the `predict`
                stage will raise a `RuntimeError`
            image_processor (Callable | None, optional):
            train_dl_kwargs (dict[str, Any] | None, optional):
            val_dl_kwargs (dict[str, Any] | None, optional):
            test_dl_kwargs (dict[str, Any] | None, optional):
            predict_dl_kwargs (dict[str, Any] | None, optional):
            cache_dir (Path | str, optional):
            classes (list | Tensor | np.ndarray | None, optional): List of
                classes to keep. For example if `classes=[1, 2]`, only those
                samples whose label is `1` or `2` will be present in the
                dataset. If `None`, all classes are kept. Note that this only
                applies to the `train` and `val` splits, the `test` and
                `predict` splits (if they exist) will not be filtered.
            label_key (str, optional): Name of the column containing the
                label. Only relevant if `classes` is not `None`.
        """

        def ds_split_factory(
            split: str, apply_filter: bool = True
        ) -> Callable[[], Dataset]:
            """
            Returns a function that loads the dataset split.

            Args:
                split (str): Name of the split, see constructor arguments
                apply_filter (bool, optional): If `True` and the constructor
                    has `classes` set, then the dataset is filtered to only
                    keep those samples whose label is in `classes`. You
                    probably want to set this to `False` for the prediction
                    split and maybe even the test split.

            Returns:
                Callable[[], Dataset]: The dataset factory
            """

            def wrapped() -> Dataset:
                ds = load_dataset(
                    dataset_name,
                    split=split,
                    cache_dir=str(cache_dir),
                    trust_remote_code=True,
                )
                if image_processor is not None:
                    ds.set_transform(image_processor)
                if apply_filter and classes:
                    ds = ds.filter(
                        lambda l: l in classes, input_columns=label_key
                    )
                ds = ds.add_column("_idx", range(len(ds)))
                return ds

            return wrapped

        super().__init__(
            train=ds_split_factory(fit_split),
            val=ds_split_factory(val_split),
            test=(ds_split_factory(test_split) if test_split else None),
            predict=(
                ds_split_factory(predict_split, False)
                if predict_split
                else None
            ),
            train_dl_kwargs=train_dl_kwargs,
            val_dl_kwargs=val_dl_kwargs,
            test_dl_kwargs=test_dl_kwargs,
            predict_dl_kwargs=predict_dl_kwargs,
        )
        self.label_key = label_key

    def _get_dataset(self, split: Literal["train", "val", "test"]) -> Dataset:
        if split not in self._datasets:
            if split == "train":
                factory = self.train
            elif split == "val":
                factory = self.val
            elif split == "test":
                factory = self.test
            else:
                raise ValueError(f"Unknown split: {split}")
            assert callable(factory)
            self._datasets[split] = factory()
        return self._datasets[split]

    def n_classes(
        self, split: Literal["train", "val", "test"] = "train"
    ) -> int:
        """
        Returns the number of classes in a given split

        Args:
            split (Literal["train", "val",
                "test"], optional): Not the true name of the split
                (as specified on the dataset's HuggingFace page), just either
                `train`, `val`, or `test`.
        """
        ds = self._get_dataset(split)
        return ds.features[self.label_key].num_classes

    def y_true(
        self, split: Literal["train", "val", "test"] = "train"
    ) -> Tensor:
        """
        Gets the vector of true labels of a given split.

        Args:
            split (Literal["train", "val",
                "test"], optional): Not the true name of the split
                (as specified on the dataset's HuggingFace page), just either
                `train`, `val`, or `test`.

        Returns:
            An `int` tensor
        """
        return Tensor(self._get_dataset(split)[self.label_key]).int()
