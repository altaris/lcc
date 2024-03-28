"""
A Hugging Face image classification dataset wrapped inside a `WrappedDataset`,
which is itself a
[`LightningDataModule`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningDataModule.html).

See also:
    https://huggingface.co/datasets?task_categories=task_categories:image-classification
"""

from typing import Any

from datasets import load_dataset

from .wrapped import WrappedDataset


class HuggingFaceDataset(WrappedDataset):
    """See module documentation"""

    def __init__(
        self,
        dataset_name: str,
        fit_split: str = "training",
        val_split: str = "validation",
        test_split: str | None = None,
        dataloader_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            fit=lambda: load_dataset(dataset_name, split=fit_split),
            val=lambda: load_dataset(dataset_name, split=val_split),
            test=(
                (lambda: load_dataset(dataset_name, split=test_split))
                if test_split
                else None
            ),
            predict=None,
            dataloader_kwargs=dataloader_kwargs,
        )
