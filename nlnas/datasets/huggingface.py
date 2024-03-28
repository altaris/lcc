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

See also:
    https://huggingface.co/datasets?task_categories=task_categories:image-classification
"""

from pathlib import Path
from typing import Any, Callable

from datasets import Dataset, load_dataset
from transformers.image_processing_utils import BaseImageProcessor

from .wrapped import WrappedDataset

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "datasets"
"""
Default download path for huggingface dataset.

See also:
    https://huggingface.co/docs/datasets/cache
"""


class HuggingFaceDataset(WrappedDataset):
    """See module documentation"""

    image_processor: BaseImageProcessor | None

    def __init__(
        self,
        dataset_name: str,
        fit_split: str = "training",
        val_split: str = "validation",
        test_split: str | None = None,
        image_processor: BaseImageProcessor | None = None,
        dataloader_kwargs: dict[str, Any] | None = None,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
    ) -> None:

        def factory(split: str) -> Callable[[], Dataset]:
            def wrapped() -> Dataset:
                d = load_dataset(
                    dataset_name, split=split, cache_dir=str(cache_dir)
                )
                d.set_transform(self.transform)
                return d

            return wrapped

        super().__init__(
            train=factory(fit_split),
            val=factory(val_split),
            test=factory(test_split) if test_split else None,
            predict=None,
            dataloader_kwargs=dataloader_kwargs,
        )
        self.image_processor = image_processor

    def transform(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Batched dataset transform"""
        if self.image_processor is None:
            return batch
        return {
            k: (
                self.image_processor(v, return_tensors="pt")["pixel_values"]
                if k in ["img", "image"]
                else v
            )
            for k, v in batch.items()
        }
