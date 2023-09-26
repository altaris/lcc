"""Custom torchvision transforms"""

from torch import Tensor


class EnsuresRGB:
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
