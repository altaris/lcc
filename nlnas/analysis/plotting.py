"""Plotting stuff"""

from pathlib import Path

import bokeh.layouts as bkl
import bokeh.plotting as bk
import numpy as np
from torch import Tensor
from tqdm import tqdm

from ..plotting import (
    class_matching_plot,
    class_scatter,
    export_png,
    make_same_xy_range,
)


def louvain_clustering_plots(
    z: np.ndarray | Tensor | list[float],
    y_true: np.ndarray | Tensor | list[int],
    y_louvain: np.ndarray | Tensor | list[int],
    matching: dict[int, set[int]],
    k: int,
    output_dir: Path,
) -> tuple:
    """
    (Used as a step in `analyse_ckpt`) Makes two plots
    1. Side-by-side scatter plots of the ground truth and Louvain communities
    2. A class matching plot, see `nlnas.plotting.class_matching_plot`

    Saves the PNGs and returns the bokeh figures in this order.

    Args:
        z (np.ndarray): An array of latent embeddings, which has shape `(N, 2)`
        y_true (np.ndarray): A `(N,)` tensor of true labels
        y_louvain (np.ndarray): A `(N,)` tensor of Louvain labels
        matching (dict[int, set[int]]): See
            `nlnas.correction.clustering.class_otm_matching`
        k (int): Number of neighbots that have been consitered when
            creating `y_louvain`. This is used in the title of the plot.
    """
    z = np.array(z)
    y_true = np.array(y_true, dtype=int)
    y_louvain = np.array(y_louvain, dtype=int)
    fig_true = bk.figure(title="Ground truth")
    class_scatter(fig_true, z, y_true, palette="viridis")
    fig_louvain = bk.figure(
        title=f"Louvain communities ({y_louvain.max() + 1}), k = {k}",
    )
    class_scatter(fig_louvain, z, y_louvain)
    make_same_xy_range(fig_true, fig_louvain)
    fig_scatter = bkl.row(fig_true, fig_louvain)
    fig_match = class_matching_plot(z, y_true, y_louvain, matching)
    export_png(fig_scatter, filename=output_dir / "scatter.png")
    export_png(fig_match, filename=output_dir / "match.png")
    return fig_scatter, fig_match


def plot_latent_samples(
    e: dict[str, np.ndarray],
    y_true: Tensor | np.ndarray | list[int],
    output_dir: Path,
) -> dict[str, bk.figure]:
    """
    (Used as a step in `analyse_ckpt`) Plots UMAP embeddings of latent samples.
    `e` should be the output of `embed_latent_samples`. Saves the PNGs and
    return the bokeh figures.

    Args:
        e (dict[str, np.ndarray]): A dict of tensors of shape `(N, 2)`
        y_true (Tensor): A true label tensor of shape `(N,)`
        output_dir (Path): The directory to save the PNGs

    Returns:
        A dict of bokeh figures. The keys are the same as `e`.
    """
    y_true, figures = np.array(y_true), {}
    progress = tqdm(e.items(), desc="UMAP plotting", leave=False)
    for k, v in progress:
        progress.set_postfix({"submodule": k})
        figure = bk.figure(title=k, toolbar_location=None)
        class_scatter(figure, v, y_true)
        figures[k] = figure
        export_png(figure, filename=output_dir / (k + ".png"))
    return figures
