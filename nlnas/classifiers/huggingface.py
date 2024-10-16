"""See `HuggingFaceClassifier` documentation."""

from typing import Any, Callable

from transformers import AutoImageProcessor, AutoModelForImageClassification

from .wrapped import WrappedClassifier


class HuggingFaceClassifier(WrappedClassifier):
    """
    Pretrained classifier model loaded from the [HuggingFace model
    hub](https://huggingface.co/models?pipeline_tag=image-classification).
    """

    def __init__(
        self,
        model_name: str,
        n_classes: int,
        head_name: str | None = None,
        **kwargs,
    ) -> None:
        """
        See also:
            `nlnas.classifiers.WrappedClassifier.__init__` and
            `nlnas.classifiers.BaseClassifier.__init__`.

        Args:
            model_name (str): Model name as in the [HuggingFace model
                hub](https://huggingface.co/models?pipeline_tag=image-classification).
                If the model name starts with `timm/`, use
                `nlnas.classifiers.TimmClassifier` instead.
            n_classes (int): See `nlnas.classifiers.WrappedClassifier.__init__`.
            head_name (str | None, optional): See
                `nlnas.classifiers.WrappedClassifier.__init__`.
        """
        if model_name.startswith("timm/"):
            raise ValueError(
                "If the model name starts with `timm/`, use "
                "`nlnas.classifiers.TimmClassifier` instead."
            )
        model = AutoModelForImageClassification.from_pretrained(model_name)
        super().__init__(model, n_classes, head_name, **kwargs)
        self.save_hyperparameters()

    @staticmethod
    def get_image_processor(
        model_name: str, **kwargs
    ) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """
        Wraps the HuggingFace `AutoImageProcessor` associated to a given model.

        Args:
            model_name (str): Model name as in the [HuggingFace model
                hub](https://huggingface.co/models?pipeline_tag=image-classification).
                Must not start with `timm/`.

        Returns:
            A callable that uses a
            [`transformers.AutoImageProcessor`](https://huggingface.co/docs/transformers/v4.44.2/en/model_doc/auto#transformers.AutoImageProcessor)
            under the hood.
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
