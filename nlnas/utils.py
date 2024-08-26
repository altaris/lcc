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
    key: str = "",
    mask: np.ndarray | Tensor | None = None,
    max_n_batches: int | None = None,
    tqdm_style: Literal["notebook", "console", "none"] | None = None,
):
    """
    Inverse of `save_tensor_batched`. The batch files should be named following
    the following pattern: `output_dir/<prefix>.<batch_idx>.<extension>`.

    Args:
        output_dir (str | Path):
        prefix (str, optional):
        extension (str, optional):
        key (str, optional): The key to use when loading the file. Batches are
            stored in safetensor files, which are essentially dictionaries. This
            arg specifies which key contains the data of interest.
        mask (np.ndarray | Tensor | None, optional): If provided, a boolean mask
            is applied on each batch. Use this if the full tensor is too large
            to fit in memory while only parts of it are actually required. The
            length if the mask should be at least the length of the full tensor.
        max_n_batches (int | None, optional): If provided, only the first
            `max_n_batches` are loaded
        tqdm_style (Literal[&quot;notebook&quot;, &quot;console&quot;, &quot;none&quot;] | None, optional):

    Returns:
        A `torch.Tensor`.
    """
    paths = list(sorted(Path(output_dir).glob(f"{prefix}.*.{extension}")))
    if max_n_batches is not None:
        paths = paths[:max_n_batches]
    n_loaded_rows = 0  # number of loaded rows BEFORE applying the mask
    batches = []
    for path in make_tqdm(tqdm_style)(paths, "Loading", leave=False):
        batch = st.load_file(path)[key]
        n_loaded_rows += batch.shape[0]
        if mask is not None:
            # TODO: need to copy the masked tensor to make sure the original one
            # is garbage collected?
            batch = batch[mask[n_loaded_rows - batch.shape[0] : n_loaded_rows]]
        batches.append(batch)
    return torch.concat(batches)


def make_tqdm(
    style: Literal["notebook", "console", "none"] | None = "console"
) -> Callable:
    """Returns the appropriate tqdm factory function based on the style"""

    def _fake_tqdm(x: Any, *_, **__):
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
    extension: str = "st",
    key: str = "",
    batch_size: int = 256,
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
        extension (str, optional):
        key (str, optional): The key to use when loading the file. Batches are
            stored in safetensor files, which are essentially dictionaries. This
            arg specifies which key contains the data of interest.
        batch_size (int, optional):
    """
    batches = (Tensor(x) if isinstance(x, np.ndarray) else x).split(batch_size)
    t = make_tqdm(tqdm_style)
    for i, batch in enumerate(t(batches, "Saving", leave=False)):
        data = {
            key: batch,
        }
        st.save_file(data, Path(output_dir) / f"{prefix}.{i:04}.{extension}")
