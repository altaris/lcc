"""Useful stuff"""

import os
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import torch
from safetensors import torch as st
from torch import Tensor, nn


def best_device() -> str:
    """Self-explanatory"""
    accelerator = os.getenv("PL_ACCELERATOR", "auto").lower()
    if accelerator == "gpu" and torch.cuda.is_available():
        return "cuda"
    if accelerator == "auto":
        return (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    return accelerator


def load_tensor_batched(
    output_dir: str | Path,
    prefix: str = "batch",
    extension: str = "st",
    tqdm_style: Literal["notebook", "console", "none"] | None = None,
):
    """Inverse of `save_tensor_batched`."""
    t = make_tqdm(tqdm_style)
    return torch.concat(
        [
            st.load_file(p)[""]
            for p in t(
                sorted(Path(output_dir).glob(f"{prefix}.*.{extension}")),
                "Loading",
                leave=False,
            )
        ]
    )


def make_tqdm(
    style: Literal["notebook", "console", "none"] | None = "console"
) -> Callable:
    """Returns the appropriate tqdm factory function based on the style"""

    def _fake_tqdm(x: Any, *args, **kwargs):  # pylint: disable=unused-argument
        return x

    if style is None or style == "none":
        f = _fake_tqdm
    elif style == "console":
        from tqdm import tqdm as f  # type: ignore
    elif style == "notebook":
        from tqdm.notebook import tqdm as f  # type: ignore
    else:
        raise ValueError(
            f"Unknown TQDM style '{style}'. Available styles are 'notebook', "
            "'console', or None"
        )
    return f


def pretty_print_submodules(
    module: nn.Module,
    exclude_non_trainable: bool = False,
    max_depth: int | None = None,
    prefix: str = "",
    current_depth: int = 0,
):
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
        exclude_non_trainable (bool, optional):
        max_depth (int | None, optional):
        prefix (str, optional): Don't use
        current_depth (int, optional): Don't use
    """
    if max_depth is not None and current_depth > max_depth:
        return
    for k, v in module.named_children():
        if exclude_non_trainable and len(list(v.parameters())) == 0:
            continue
        print(prefix + k, "->", v.__class__.__name__)
        p = prefix.replace("-", " ") + "|" + ("-" * len(k))
        pretty_print_submodules(
            module=v,
            exclude_non_trainable=exclude_non_trainable,
            max_depth=max_depth,
            prefix=p,
            current_depth=current_depth + 1,
        )


def save_tensor_batched(
    x: Tensor | np.ndarray,
    output_dir: str | Path,
    prefix: str = "batch",
    batch_size: int = 256,
    extension: str = "st",
    tqdm_style: Literal["notebook", "console", "none"] | None = None,
) -> None:
    """
    Saves a tensor in batches of `batch_size` elements. The files will be named
    `output_dir/<prefix>.<batch_idx>.<extension>`. The batches are saved using
    safetensors.

    It would be great if you could adjust the batch size so that there are less
    than 10000 batches :]

    Args:
        x (Tensor):
        output_dir (str):
        prefix (str, optional):
        batch_size (int, optional):
        extension (str, optional):
    """
    batches = (Tensor(x) if isinstance(x, np.ndarray) else x).split(batch_size)
    t = make_tqdm(tqdm_style)
    for i, batch in enumerate(t(batches, "Saving", leave=False)):
        st.save_file(
            {"": batch}, Path(output_dir) / f"{prefix}.{i:04}.{extension}"
        )
