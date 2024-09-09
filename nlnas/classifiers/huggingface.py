"""
Pretrained classifier model loaded from the [HuggingFace model
hub](https://huggingface.co/models?pipeline_tag=image-classification).
"""

from typing import Any, Callable

from transformers import AutoImageProcessor, AutoModelForImageClassification

from .wrapped import WrappedClassifier


class HuggingFaceClassifier(WrappedClassifier):
    """See module documentation."""

    def __init__(
        self,
        model_name: str,
        n_classes: int,
        head_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        model = AutoModelForImageClassification.from_pretrained(model_name)
        super().__init__(model, n_classes, head_name, **kwargs)
        self.save_hyperparameters()

    @staticmethod
    def get_image_processor(model_name: str, **kwargs) -> Callable:
        """
        Wraps the HuggingFace `AutoImageProcessor` associated to a given model
        """
        hf_transorm = AutoImageProcessor.from_pretrained(model_name)

        def _transform(batch: dict[str, Any]) -> dict[str, Any]:
            return {
                k: (
                    hf_transorm(
                        [img.convert("RGB") for img in v], return_tensors="pt"
                    )["pixel_values"]
                    if k in ["img", "image"]  # TODO: pass image_key
                    else v
                )
                for k, v in batch.items()
            }

        return _transform
