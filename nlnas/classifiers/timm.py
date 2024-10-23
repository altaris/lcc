"""See `TimmClassifier` documentation."""

from typing import Any, Callable

import timm
from torch import Tensor

from .wrapped import Batch, WrappedClassifier


class TimmClassifier(WrappedClassifier):
    """
    Pretrained classifier model loaded from the [HuggingFace model
    hub](https://huggingface.co/models?pipeline_tag=image-classification)
    uploaded by the `timm` team.
    """

    image_transform: Callable

    def __init__(
        self,
        model_name: str,
        n_classes: int,
        head_name: str | None = None,
        pretrained: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        See also:
            `nlnas.classifiers.WrappedClassifier.__init__` and
            `nlnas.classifiers.BaseClassifier.__init__`.

        Args:
            model_name (str): Model name as in the [HuggingFace model
                hub](https://huggingface.co/models?pipeline_tag=image-classification).
                Must start with `timm/`.
            n_classes (int): See `nlnas.classifiers.WrappedClassifier.__init__`.
            head_name (str | None, optional): See
                `nlnas.classifiers.WrappedClassifier.__init__`.
            pretrained (bool, optional): Defaults to `True`.
        """
        if not model_name.startswith("timm/"):
            raise ValueError(
                "The model isn't a timm model (its name does not start with "
                "`timm/`). Use `nlnas.classifiers.HuggingFaceClassifier` "
                "instead."
            )
        model = timm.create_model(model_name, pretrained=pretrained)
        super().__init__(model, n_classes, head_name, **kwargs)
        self.save_hyperparameters()

    @staticmethod
    def get_image_processor(model_name: str, **kwargs: Any) -> Callable:
        """
        Wraps the HuggingFace `AutoImageProcessor` associated to a given model.

        See also:
            [`timm.create_transform`](https://huggingface.co/docs/timm/reference/data#timm.data.create_transform).

        Args:
            model_name (str): Model name as in the [HuggingFace model
                hub](https://huggingface.co/models?pipeline_tag=image-classification).
                Must start with `timm/`.

        Returns:
            A callable that uses a [Torchvision
            transform](https://pytorch.org/vision/0.19/transforms.html) under
            the hood.
        """
        model = timm.create_model(model_name, pretrained=False)
        conf = timm.data.resolve_model_data_config(model)
        conf["is_training"], conf["no_aug"] = True, True
        timm_transform = timm.data.create_transform(**conf)

        def _transform(batch: dict[str, Any]) -> dict[str, Any]:
            return {
                k: (
                    (
                        [timm_transform(img.convert("RGB")) for img in v]
                        if isinstance(v, list)
                        else timm_transform(v)
                    )
                    if k in ["img", "image"]  # TODO: pass image_key from DS
                    else v
                )
                for k, v in batch.items()
            }

        return _transform

    def forward(self, inputs: Tensor | Batch, *_: Any, **__: Any) -> Tensor:
        x: Tensor = (
            inputs if isinstance(inputs, Tensor) else inputs[self.image_key]
        )
        return self.model(x.to(self.device))  # type: ignore
