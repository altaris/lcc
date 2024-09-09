"""Distance distribution stuff"""

from itertools import combinations_with_replacement, product
from math import ceil

import bokeh.plotting as bk
import numpy as np
from joblib import Parallel, delayed
from loguru import logger as logging
from scipy.spatial.distance import cdist
from scipy.stats import chi


def _batch_dh(
    b1: np.ndarray,
    b2: np.ndarray,
    resolution: int = 500,
    interval: tuple[float, float] = (0.0, 2.5),
) -> np.ndarray:
    """
    Computes the distance histogram between two batches
    """
    c = cdist(b1, b2, metric="euclidean") / np.sqrt(b1.shape[-1])
    h, _ = np.histogram(c, bins=resolution, range=interval, density=False)
    return h


def distance_distribution(
    a: np.ndarray,
    b: np.ndarray | None = None,
    batch_size: int = 1024,
    resolution: int = 500,
    interval: tuple[float, float] = (0.0, 2.5),
    n_jobs: int = 32,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the normalized Euclidean distance distribution of a 2D array.

    Let's say `x` has shape `(N, d)`. Since `N` can be large, this methods cuts
    `x` in batches and compute the distance histogram (aka unnormalized
    distribution) between batch pairs independently. Then all histograms are
    coalesced into a single distance distribution.

    This method uses joblib to compute the batch pair distance histograms.

    Returns:
        A normalized hisogram. The first array has shape `(resolution,)` and
        represents the bin densities, and the second has shape
        `(resolution + 1,)` and contains the bin edges. Furthermore,
        `edges[0] == interval[0]` and `edges[-1] == edges[resolution] ==
        interval[1]`.
    """
    f = lambda t: _batch_dh(
        t[0], t[1], resolution=resolution, interval=interval
    )
    if b is None:
        ba = np.array_split(a, ceil(a.shape[0] / batch_size), axis=0)
        jobs = map(delayed(f), combinations_with_replacement(ba, 2))
        nj = len(ba) * (len(ba) - 1) + len(ba)
        logging.debug("Number of tasks: {}", nj)
    else:
        ba = np.array_split(a, ceil(a.shape[0] / batch_size), axis=0)
        bb = np.array_split(b, ceil(b.shape[0] / batch_size), axis=0)
        jobs = map(delayed(f), product(ba, bb))
        nj = len(ba) * len(bb)
        logging.debug("Number of tasks: {}", nj)
    if nj <= 1 or n_jobs == 1:
        results = [j[0](*j[1], **j[2]) for j in jobs]
    else:
        executor = Parallel(n_jobs=n_jobs, verbose=1 if verbose else 0)
        results = executor(jobs)
    hist = np.stack(results).sum(axis=0)
    hist = hist.astype(float) / hist.sum()
    # dividing by the sum gives a density since all bins have equal width
    edges = np.linspace(
        start=interval[0], stop=interval[1], num=resolution + 1, endpoint=True
    )
    return hist, edges


def distance_distribution_plot(
    hist: np.ndarray,
    edges: np.ndarray,
    n_dims: int | None = None,
    height: int = 500,
    x_range: tuple[int, int] | None = None,
    y_range: tuple[int, int] | None = None,
    include_chi: bool = True,
) -> bk.figure:
    """
    data can either be a distance matrix or a histogram-edge array pair as
    returned by np.histogram. In the first scenarion the `resolution` argument
    is ignored.

    Args:
        hist (np.ndarray):
        edges (np.ndarray):
        n_dims (int, optional):
        height (int, optional): The final plot will have width `2 * height`.
        x_range (tuple[int, int] | None, optional):
        y_range (tuple[int, int] | None, optional):
        include_chi (bool, optional): Whether to include the chi distribution
            in the plot. If set to `True`, `n_dims` must be provided.

    Returns:
        A bokeh figure.
    """
    figure = bk.figure(
        height=height,
        width=2 * height,
        y_range=y_range or (0, 1.1 * hist.max()),
        x_range=x_range or (0, 2.5),
        toolbar_location=None,
    )
    figure.line(edges[:-1], hist, color="black", width=2)
    if include_chi:
        if n_dims is None:
            raise ValueError("n_dims must be provided if include_chi is True")
        pdf = lambda x: chi.pdf(x, df=n_dims, scale=np.sqrt(2 / n_dims))
        a = np.linspace(edges[0], edges[-1], len(hist))
        b = np.array([pdf(x) for x in a])
        figure.line(a, b, color="grey", width=0.5)
    return figure
