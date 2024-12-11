"""Useful stuff."""

import os
from functools import partial
from typing import Any, Callable, Literal, TypeAlias

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import Tensor, nn
from tqdm import tqdm

TqdmStyle: TypeAlias = Literal["notebook", "console", "none"] | None
"""Type alias for supported TQDM styles of `make_tqdm`."""


def check_cuda(
    requested_device: Any = None,
) -> tuple[bool, Any]:
    """
    Tries to accomodate the requested device. Returns a boolean indicating
    wether CUDA is requested and available, as well as a device string that may
    or may not be the same as the requested one.

    If the requested device is `None`, the cuda is used if available.
    """
    use_cuda = (
        requested_device == "cuda" or requested_device is None
    ) and torch.cuda.is_available()
    if use_cuda:
        device = "cuda"
    elif requested_device is None:
        device = "cpu"
    else:
        device = requested_device
    return use_cuda, device


def get_reasonable_n_jobs() -> int:
    """
    Gets a reasonable number of jobs for parallel processing. Reasonable means
    it's not going to slam your system (hopefully). See the implementation for
    the exact scheme.
    """
    n = os.cpu_count()
    if n is None:
        return 1
    if n <= 8:
        return n // 2
    return int(n * 2 / 3)


def make_tqdm(style: TqdmStyle = "console") -> Callable[..., tqdm]:
    """
    Returns the appropriate tqdm factory function based on the style.

    Args:
        style (TqdmStyle, optional): Defaults to "console".
    """

    if style is None or style == "none":
        from tqdm import tqdm as _tqdm

        return partial(_tqdm, disable=True, leave=False)
    if style == "console":
        from tqdm import tqdm as _tqdm

        return partial(_tqdm, leave=False)
    if style == "notebook":
        from tqdm.notebook import tqdm as _tqdm

        return partial(_tqdm, leave=False)
    raise ValueError(
        f"Unknown TQDM style '{style}'. Available styles are 'notebook', "
        "'console', or None"
    )


def pretty_print_submodules(
    module: nn.Module,
    exclude_non_trainable: bool = False,
    max_depth: int | None = None,
    _prefix: str = "",
    _current_depth: int = 0,
) -> None:
    """
    Recursively prints a module and its submodule in a hierarchical manner.

        >>> pretty_print_submodules(model, max_depth=4)
        model -> ResNetForImageClassification
        |-----resnet -> ResNetModel
        |     |------embedder -> ResNetEmbeddings
        |     |      |--------embedder -> ResNetConvLayer
        |     |      |        |--------convolution -> Conv2d
        |     |      |        |--------normalization -> BatchNorm2d
        |     |      |        |--------activation -> ReLU
        |     |      |--------pooler -> MaxPool2d
        |     |------encoder -> ResNetEncoder
        |     |      |-------stages -> ModuleList
        |     |      |       |------0 -> ResNetStage
        |     |      |       |------1 -> ResNetStage
        |     |      |       |------2 -> ResNetStage
        |     |      |       |------3 -> ResNetStage
        |     |------pooler -> AdaptiveAvgPool2d
        |-----classifier -> Sequential
        |     |----------0 -> Flatten
        |     |----------1 -> Linear

    Args:
        module (nn.Module):
        exclude_non_trainable (bool, optional): Defaults to `True`.
        max_depth (int | None, optional): Defaults to `None`, which means no
            depth cap.
        _prefix (str, optional): Don't use.
        _current_depth (int, optional): Don't use.
    """
    if max_depth is not None and _current_depth > max_depth:
        return
    for k, v in module.named_children():
        if exclude_non_trainable and len(list(v.parameters())) == 0:
            continue
        print(_prefix + k, "->", v.__class__.__name__)
        p = _prefix.replace("-", " ") + "|" + ("-" * len(k))
        pretty_print_submodules(
            module=v,
            exclude_non_trainable=exclude_non_trainable,
            max_depth=max_depth,
            _prefix=p,
            _current_depth=_current_depth + 1,
        )


def to_array(x: ArrayLike, **kwargs: Any) -> np.ndarray:
    """
    Converts an array-like object to a numpy array. If the input is a tensor,
    it is detached and moved to the CPU first.
    """
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return x if isinstance(x, np.ndarray) else np.array(x, **kwargs)


def to_int_array(x: ArrayLike, **kwargs: Any) -> np.ndarray:
    """
    Converts an array-like object to a numpy array of integers. If the input is
    a tensor, it is detached and moved to the CPU first.
    """
    return to_array(x, **kwargs).astype(int)


def to_int_tensor(x: ArrayLike, **kwargs: Any) -> Tensor:
    """
    Converts an array-like object to a torch tensor of integers. If the input is
    already a tensor of integers, it is returned as is. If `x` is a tensor but
    not of integers, it is just converted to integers.
    """
    if isinstance(x, Tensor) and x.dtype == torch.int:
        return x
    return to_tensor(x, **kwargs).to(dtype=torch.int)


def to_tensor(x: ArrayLike, **kwargs: Any) -> Tensor:
    """
    Converts an array-like object to a torch tensor. If `x` is already a tensor,
    then it is returned as is. In particular, if `x` is a tensor this method is
    differentiable and its derivative is the identity function.
    """
    return x if isinstance(x, Tensor) else torch.tensor(x, **kwargs)
