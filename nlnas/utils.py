"""Useful stuff."""

import os
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

import numpy as np
import torch
from safetensors import torch as st
from torch import Tensor, nn


def load_tensor_batched(
    output_dir: str | Path,
    prefix: str = "batch",
    extension: str = "st",
    key: str = "",
    mask: np.ndarray | Tensor | None = None,
    max_n_batches: int | None = None,
    tqdm_style: Literal["notebook", "console", "none"] | None = None,
    device: Literal["cpu", "cuda"] | None = None,
) -> Tensor:
    """
    Inverse of `save_tensor_batched`. The batch files should be named following
    the following pattern:

        output_dir/<prefix>.<batch_idx>.<extension>

    Args:
        output_dir (str | Path):
        prefix (str, optional):
        extension (str, optional): Extension without the first `.`. Defaults to
            `st`.
        key (str, optional): The key to use when loading the file. Batches are
            stored in safetensor files, which are essentially dictionaries. This
            arg specifies which key contains the data of interest. Defaults to
            `""`, which is the key used in `nlnas.utils.save_tensor_batched` and
            `nlnas.classifiers.full_dataset_evaluation`.
        mask (np.ndarray | Tensor | None, optional): If provided, a boolean mask
            is applied on each batch. Use this if the full tensor is too large
            to fit in memory while only parts of it are actually required. The
            length if the mask should be at least the length of the full tensor.
        max_n_batches (int | None, optional): If provided, only the first
            `max_n_batches` are loaded
        tqdm_style (Literal["notebook", "console", "none"] | None, optional):
            Progress bar style.

    Returns:
        A `torch.Tensor`.
    """
    iterator = load_tensor_batched_iter(
        output_dir=output_dir,
        prefix=prefix,
        extension=extension,
        key=key,
        mask=mask,
        max_n_batches=max_n_batches,
        device=device,
    )
    if tqdm_style is None or tqdm_style == "none":
        torch.concat(list(iterator))
    paths = list(sorted(Path(output_dir).glob(f"{prefix}.*.{extension}")))
    if max_n_batches is not None:
        paths = paths[:max_n_batches]
    n_batches, batches = len(paths), []
    # All this circus is to get a nice progress bar
    for _ in make_tqdm(tqdm_style)(range(n_batches), "Loading", leave=False):
        batches.append(next(iterator))
    return torch.concat(batches)


def load_tensor_batched_iter(
    output_dir: str | Path,
    prefix: str = "batch",
    extension: str = "st",
    key: str = "",
    mask: np.ndarray | Tensor | None = None,
    max_n_batches: int | None = None,
    device: Literal["cpu", "cuda"] | None = None,
) -> Iterator[Tensor]:
    """
    Same as `nlnas.utils.load_tensor_batched`, but returns an iterator over the
    batches rather than all batches concatenated into a single tensor. See
    `nlnas.utils.load_tensor_batched` for the meaning of the arguments.

    Warning:
        If `mask` is provided, the iterator may yield batches of different lengths.

    Args:
        output_dir (str | Path):
        prefix (str, optional):
        extension (str, optional):
        key (str, optional):
        mask (np.ndarray | Tensor | None, optional):
        max_n_batches (int | None, optional):
        device (Literal['cpu', 'cuda'] | None, optional):

    Returns:
        Iterator[Tensor]
    """
    paths = list(sorted(Path(output_dir).glob(f"{prefix}.*.{extension}")))
    if max_n_batches is not None:
        paths = paths[:max_n_batches]
    n_loaded_rows = 0  # number of loaded rows BEFORE applying the mask
    for path in paths:
        batch = st.load_file(path)[key]
        n_loaded_rows += batch.shape[0]
        if not (mask is None or mask.all()):
            # TODO: need to copy the masked tensor to make sure the original one
            # is garbage collected?
            batch = batch[mask[n_loaded_rows - batch.shape[0] : n_loaded_rows]]
        yield batch.to(device)


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


def make_tqdm(
    style: Literal["notebook", "console", "none"] | None = "console",
) -> Callable:
    """
    Returns the appropriate tqdm factory function based on the style. Note that
    if the style is `"none"` or `None`, a fake tqdm function is returned that is
    just the identity function. In particular, the usual `tqdm` methods like
    `set_postfix` cannot be used.

    Args:
        style (Literal['notebook', 'console', 'none'] | None, optional):
            Defaults to "console".
    """

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
    _prefix: str = "",
    _current_depth: int = 0,
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


def save_tensor_batched(
    x: Tensor | np.ndarray | list,
    output_dir: str | Path,
    prefix: str = "batch",
    extension: str = "st",
    key: str = "",
    batch_size: int = 1024,
    tqdm_style: Literal["notebook", "console", "none"] | None = None,
) -> None:
    """
    Saves a tensor in batches of `batch_size` elements. The files will be named

        output_dir/<prefix>.<batch_idx>.<extension>

    The batches are saved using
    [Safetensors](https://huggingface.co/docs/safetensors/index).

    The `batch_idx` string is 4 digits long, so would be great if you could
    adjust the batch size so that there are less than 10000 batches :]

    Args:
        x (Tensor | np.ndarray | list):
        output_dir (str):
        prefix (str, optional):
        extension (str, optional): Without the first `.`. Defaults to `st`.
        key (str, optional): The key to use when saving the file. Batches are
            stored in safetensor files, which are essentially dictionaries. This
            arg specifies which key contains the data of interest.
        batch_size (int, optional): Defaults to $1024$.
        tqdm_style (Literal["notebook", "console", "none"] | None, optional):
            Progress bar style.
    """
    batches = to_tensor(x).split(batch_size)
    t = make_tqdm(tqdm_style)
    for i, batch in enumerate(t(batches, "Saving", leave=False)):
        data = {
            key: batch,
        }
        st.save_file(data, Path(output_dir) / f"{prefix}.{i:04}.{extension}")


def to_array(x: Any, **kwargs) -> np.ndarray:
    """
    Converts an array-like object to a numpy array. If the input is a tensor,
    it is detached and moved to the CPU first
    """
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return x if isinstance(x, np.ndarray) else np.array(x, **kwargs)


def to_tensor(x: Any, **kwargs) -> Tensor:
    """
    Converts an array-like object to a torch tensor. If `x` is already a tensor,
    then it is returned as is. In particular, if `x` is a tensor this method is
    differentiable and its derivative is the identity function.
    """
    return x if isinstance(x, Tensor) else torch.tensor(x, **kwargs)
