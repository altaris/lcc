"""Plotting utilities"""

import bokeh.layouts as bkl
import bokeh.models as bkm
import bokeh.palettes as bkp
import bokeh.plotting as bk
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

from .correction import otm_matching_predicates

BK_PALETTE_FUNCTIONS = {
    "cividis": bkp.cividis,
    "gray": bkp.gray,
    "grey": bkp.grey,
    "inferno": bkp.inferno,
    "magma": bkp.magma,
    "viridis": bkp.viridis,
}


def class_scatter(
    plot: bk.figure,
    x: np.ndarray,
    y: np.ndarray,
    palette: bkp.Palette | list[str] | str | None = None,
    size: float = 3,
) -> None:
    """
    Scatter plot where each class has a different color. Points in negative
    classes (those for which the `y` value is strictly less than 0), called
    _outliers_ here, are all plotted black.

    Example:

        ![Example 1](../docs/imgs/class_scatter.png)

        (this example does't have outliers but I'm sure you can use your
        imagination)

    Args:
        plot (bk.figure):
        x (np.ndarray): A `(N, 2)` array
        y (np.ndarray): A `(N,)` int array. Each unique value corresponds to a
            class
        palette (Palette | list[str] | str | None, optional): Either a
            palette object (see
            https://docs.bokeh.org/en/latest/docs/reference/palettes.html#bokeh-palettes),
            a list of HTML colors (at least as many as the number of classes),
            or a name in `nlnas.plotting.BK_PALETTE_FUNCTIONS`.
        size (float, optional): Dot size. The outlier's dot size will be half
            that

    Raises:
        `ValueError` if the palette is unknown
    """
    n_classes = min(len(np.unique(y[y >= 0])), 256)
    if palette is None:
        palette = bkp.viridis(n_classes)
    if isinstance(palette, str):
        if palette not in BK_PALETTE_FUNCTIONS:
            raise ValueError(f"Unknown palette '{palette}'")
        palette = BK_PALETTE_FUNCTIONS[palette](n_classes)
    for i, j in enumerate(np.unique(y[y >= 0])[:n_classes]):
        a = x[y == j]
        plot.scatter(
            a[:, 0],
            a[:, 1],
            color=palette[i],
            line_width=0,
            size=size,
        )
    a = x[y < 0]
    plot.scatter(
        a[:, 0],
        a[:, 1],
        color="black",
        line_width=0,
        size=size / 2,
    )


# pylint: disable=too-many-locals
def gaussian_mixture_plot(
    plot: bk.figure,
    gm: GaussianMixture,
    line_color="black",
    means_color="red",
    x_min: float = -100,
    x_max: float = 100,
    y_min: float = -100,
    y_max: float = 100,
    resolution: int = 200,
    n_levels: int = 100,
) -> None:
    """
    A countour plot of a gaussian mixture density. No filling.

    Example:

    ```py
    import bokeh.plotting as bk
    plot = bk.figure()
    gaussian_mixture_plot(plot, gm)
    bk.show(plot)
    ```

    Example:
        It is possible to plot more than 1 GM on a single figure:

        ![Example 1](../docs/imgs/gaussian_mixture_plot.1.png)

        The red and blue contours each correspond to a call of
        `gaussian_mixture_plot`. Don't worry about the lines. Look I was too
        lazy to make a clean example so I just grabbed this image from a
        presentation I made in the past.

    """
    x, y = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    xy = np.stack([x, y], axis=-1).reshape(-1, 2)
    lvls = np.linspace(gm.weights_.min() / 2, 1, n_levels)

    ds = []
    for i in range(gm.n_components):
        mu, cov, w = gm.means_[i], gm.covariances_[i], gm.weights_[i]
        pdf = multivariate_normal(mu, cov).pdf
        d = pdf(xy).reshape(resolution, resolution)
        d = d / d.max() * w
        ds.append(d)

    z = np.sum(ds, axis=0)
    plot.contour(x, y, z, lvls, line_color=line_color)
    m = gm.means_
    plot.cross(m[:, 0], m[:, 1], color=means_color)


def class_matching_plot(
    x: np.ndarray,
    y_a: np.ndarray,
    y_b: np.ndarray,
    matching: dict[int, set[int]] | dict[str, set[int]],
    size: int = 400,
) -> bkm.GridBox:
    """
    Given a dataset `x` and two labellings `y_a` and `y_b`, this method makes a
    scatter plot detailling the situation. Labels in `y_a` are considered to be
    ground truth.

    Example:
        ![Example 1](../docs/imgs/class_matching_plot.png)

    Args:
        x: (np.ndarray): A `(N, 2)` array
        y_a (np.ndarray): A `(N,)` integer array with values in $\\\\{ 0, 1, ...,
            c_a - 1 \\\\}$ for some $c_a > 0$.
        y_b (np.ndarray): A `(N,)` integer array with values in $\\\\{ 0, 1, ...,
            c_b - 1 \\\\}$ for some $c_b > 0$.
        matching (dict[int, set[int]] | dict[str, set[int]]): Matching between
            the labels of `y_a` and the labels of `y_b`. If some keys are
            strings, they must be convertible to ints.
        size (int, optional): The size of each scatter plot
    """
    m = {int(k): v for k, v in matching.items()}
    p1, p2, p3, p4 = otm_matching_predicates(y_a, y_b, m)
    n_true, n_matched = p1.sum(axis=1), p2.sum(axis=1)
    n_inter = (p1 & p2).sum(axis=1)
    n_miss, n_exc = p3.sum(axis=1), p4.sum(axis=1)

    figures = []
    for a, bs in m.items():
        n_true, n_matched = p1[a].sum(), p2[a].sum()
        n_inter = (p1[a] & p2[a]).sum()
        n_miss, n_exc = p3[a].sum(), p4[a].sum()
        fig_a = bk.figure(
            width=size,
            height=size,
            title=f"Ground truth, class {a}; n = {n_true}",
        )
        class_scatter(fig_a, x[p1[a]], y_a[p1[a]])
        fig_b = bk.figure(
            width=size,
            height=size,
            title=(
                f"{len(bs)} matched classes: "
                + ", ".join(map(str, bs))
                + f"; n = {n_matched}"
            ),
        )
        class_scatter(fig_b, x[p2[a]], y_b[p2[a]])
        fig_match = bk.figure(
            width=size,
            height=size,
            title=f"Intersection; n = {n_inter}",
        )
        class_scatter(fig_match, x[p1[a] & p2[a]], y_b[p1[a] & p2[a]])
        y_diff = p3[a] + 2 * p4[a]
        fig_diff = bk.figure(
            width=size,
            height=size,
            title=(
                f"Symmetric difference; n = {n_miss + n_exc}\n"
                f"Misses (red) = {n_miss}; excess (blue) = {n_exc}"
            ),
        )
        class_scatter(
            fig_diff,
            x[y_diff > 0],
            y_diff[y_diff > 0],
            palette=["#ff0000", "#0000ff"],
        )
        make_same_xy_range(fig_a, fig_b, fig_match, fig_diff)
        figures.append([fig_a, fig_b, fig_match, fig_diff])

    return bkl.grid(figures)  # type: ignore


def make_same_xy_range(*args: bk.figure) -> None:
    """Makes sure all figures share the same `x_range` and `y_range`"""
    for f in args[1:]:
        f.x_range, f.y_range = args[0].x_range, args[0].y_range
