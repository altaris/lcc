"""See `TorchvisionClassifier` documentation."""

from typing import Any, Callable

from torchvision.models import get_model, get_model_weights
from torchvision.transforms import v2 as tr

from .wrapped import WrappedClassifier


class TorchvisionClassifier(WrappedClassifier):
    """
    A torchvision classifier wrapped as a `WrappedClassifier`. See
    https://pytorch.org/vision/stable/models.html#classification .
    """

    def __init__(
        self,
        model_name: str,
        n_classes: int,
        head_name: str | None = None,
        weights: Any = "DEFAULT",
        **kwargs: Any,
    ) -> None:
        model = get_model(model_name, weights=weights)
        super().__init__(model, n_classes, head_name, **kwargs)
        self.save_hyperparameters()

    @staticmethod
    def get_image_processor(
        model_name: str, weights: str = "DEFAULT", **__: Any
    ) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """
        Creates an image processor based on the transform object of the model's
        chosen weights. For example,

            TorchvisionClassifier.get_image_processor("alexnet")

        is analogous to

            get_model_weights("alexnet")["DEFAULT"].transforms()
        """

        transform = tr.Compose(
            [tr.RGB(), get_model_weights(model_name)[weights].transforms()]
        )

        def _processor(batch: dict[str, Any]) -> dict[str, Any]:
            return {
                k: (
                    [transform(img) for img in v]
                    # TODO: pass image_key from DS ↓
                    if k in ["img", "image", "jpg", "png"]
                    else v
                )
                for k, v in batch.items()
            }

        return _processor
