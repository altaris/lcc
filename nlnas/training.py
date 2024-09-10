"""General model training utilities"""

from pathlib import Path
from typing import Literal

import pandas as pd
import regex as re


class NoCheckpointFound(Exception):
    """Raised by `nlnas.utils.last_checkpoint_path` if no checkpoint is found"""


def all_checkpoint_paths(output_path: str | Path) -> list[Path]:
    """
    Returns the sorted (by epoch) list of all checkpoints.

    Args:
        output_path (str | Path): e.g.
            `out.local/ft/cifar100/microsoft-resnet-18`

    Raises:
        NoCheckpointFound: If no checkpoint is found
    """
    r, d = re.compile(r"/epoch=(\d+)-step=\d+\.ckpt$"), {}
    for p in Path(output_path).glob("**/*.ckpt"):
        if m := re.search(r, str(p)):
            epoch = int(m.group(1))
            d[epoch] = p
    ckpts = [d[i] for i in sorted(list(d.keys()))]
    if not ckpts:
        raise NoCheckpointFound
    return ckpts


def best_checkpoint_path(
    output_path: str | Path,
    metric: str = "val/acc",
    mode: Literal["min", "max"] = "max",
) -> tuple[Path, int]:
    """
    Returns the path to the best checkpoint

    Args:
        output_path (str | Path): e.g.
            `out.local/ft/cifar100/microsoft-resnet-18`
        metric (str, optional):
        mode (Literal["min", "max"], optional):

    Returns:
        tuple[Path, int]: _description_
    """
    if not isinstance(output_path, Path):
        output_path = Path(output_path)
    ckpts = all_checkpoint_paths(output_path)
    metrics_path = list(output_path.glob("**/csv_logs/**/metrics.csv"))[0]
    epoch = best_epoch(metrics_path, metric, mode)
    return ckpts[epoch], epoch


def best_epoch(
    metrics_path: str | Path,
    metric: str = "val/acc",
    mode: Literal["min", "max"] = "max",
) -> int:
    """Given the `metrics.csv` path, returns the best epoch index"""
    df = pd.read_csv(metrics_path)
    df.drop(columns=["train/loss"], inplace=True)
    df = df.groupby("epoch").tail(1)
    df.reset_index(inplace=True, drop=True)
    return int(df[metric].argmax() if mode == "max" else df[metric].argmin())


def checkpoint_ves(path: str | Path) -> tuple[int, int, int]:
    """
    Given a checkpoint path that looks like e.g.

        out/resnet18/cifar10/model/tb_logs/resnet18/version_2/checkpoints/epoch=32-step=5181.ckpt

    returns the **v**ersion number (2), the number of **e**pochs (32), and the
    number of **s**teps (5181).
    """
    r = r".*version_(\d+)/checkpoints/epoch=(\d+)-step=(\d+)\.ckpt"
    if m := re.match(r, str(path)):
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    raise ValueError(f"Path '{path}' is not a valid checkpoint path")


def last_checkpoint_path(output_path: Path) -> Path:
    """
    Finds the file path of the last Pytorch Lightning training checkpoint
    (`ckpt` file) in a given directory. The step count is considered, rather
    than the epoch count.

    Args:
        output_path (str | Path): e.g.
            `out.local/ft/cifar100/microsoft-resnet-18`
    """
    d = {}
    for c in output_path.glob("**/*step=*.ckpt"):
        try:
            d[checkpoint_ves(c)[2]] = c
        except ValueError:
            pass
    if ks := list(d.keys()):
        return Path(d[max(ks)])
    raise NoCheckpointFound
