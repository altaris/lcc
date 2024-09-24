"""Plotting utilities"""

from pathlib import Path
from typing import Any

import bokeh.layouts as bkl
import bokeh.models as bkm
import bokeh.palettes as bkp
import bokeh.plotting as bk
import numpy as np
from loguru import logger as logging
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler

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
    figure: bk.figure,
    x: np.ndarray,
    y: np.ndarray,
    palette: bkp.Palette | list[str] | str | None = None,
    size: float = 3,
    rescale: bool = True,
    axis_visible: bool = False,
    grid_visible: bool = True,
    outliers: bool = True,
) -> None:
    """
    Scatter plot where each class has a different color. Points in negative
    classes (those for which the `y` value is strictly less than 0), called
    _outliers_ here, are all plotted black.

    Example:

        ![Example 1](../docs/imgs/class_scatter.png)

        (this example does't have outliers but I'm sure you can use your
        imagination)

    Warning:
        This method hides the figure's axis and grid lines.

    Args:
        figure (bk.figure):
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
        rescale (bool, optional): Whether to rescale the `x` values to `[0, 1]`
        outliers (bool, optional): Whether to plot the outliers (those samples
            with a label < 0)

    Raises:
        `ValueError` if the palette is unknown
    """
    if rescale:
        x = MinMaxScaler().fit_transform(x)
    n_classes = min(len(np.unique(y[y >= 0])), 256)
    if palette is None:
        palette = bkp.viridis(n_classes)
    if isinstance(palette, str):
        if palette not in BK_PALETTE_FUNCTIONS:
            raise ValueError(f"Unknown palette '{palette}'")
        palette = BK_PALETTE_FUNCTIONS[palette](n_classes)
    for i, j in enumerate(np.unique(y[y >= 0])[:n_classes]):
        if not (y == j).any():
            continue
        a = x[y == j]
        figure.scatter(
            a[:, 0],
            a[:, 1],
            color=palette[i],
            line_width=0,
            size=size,
        )
    if (y < 0).any() and outliers:
        a = x[y < 0]
        figure.scatter(
            a[:, 0],
            a[:, 1],
            color="black",
            line_width=0,
            size=size / 2,
        )
    figure.axis.visible = axis_visible
    figure.xgrid.visible = figure.ygrid.visible = grid_visible


def export_png(obj: Any, filename: str | Path) -> Path:
    """
    A replacement for `bokeh.io.export_png` which can sometimes be a bit buggy.
    Instanciates its own Firefox webdriver. A bit slower but more reliable.

    If Selenium is not installed, or if the Firefox webdriver is not installed,
    or if any other error occurs, this method will fall back to the default
    bokeh implementation.
    """
    from bokeh.io import export_png as _export_png

    webdriver: Any = None
    try:
        from selenium.webdriver import Firefox, FirefoxOptions

        opts = FirefoxOptions()
        opts.add_argument("--headless")
        webdriver = Firefox(options=opts)
        _export_png(obj, filename=str(filename), webdriver=webdriver)
    except Exception as e:
        if isinstance(e, ModuleNotFoundError):
            logging.error(
                "Selenium is not installed. Falling back to default bokeh "
                "implementation"
            )
        else:
            logging.error(
                f"Failed to export PNG using explicit Selenium driver: {e}\n"
                "Falling back to default bokeh implementation"
            )
        _export_png(obj, filename=str(filename))
    finally:
        if webdriver is not None:
            webdriver.close()  # type: ignore
    return Path(filename)


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

    Warning:
        The array `x` is rescaled to fit in the `[0, 1]` range.

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
    x = MinMaxScaler().fit_transform(x)
    m = {int(k): v for k, v in matching.items()}
    p1, p2, p3, p4 = otm_matching_predicates(y_a, y_b, m)
    n_true, n_matched = p1.sum(axis=1), p2.sum(axis=1)
    n_inter = (p1 & p2).sum(axis=1)
    n_miss, n_exc = p3.sum(axis=1), p4.sum(axis=1)

    figures = []
    kw = {
        "width": size,
        "height": size,
        "x_range": (-0.04, 1.04),
        "y_range": (-0.04, 1.04),
    }
    for a, bs in m.items():
        n_true, n_matched = p1[a].sum(), p2[a].sum()
        n_inter = (p1[a] & p2[a]).sum()
        n_miss, n_exc = p3[a].sum(), p4[a].sum()
        fig_a = bk.figure(title=f"Ground truth, class {a}; n = {n_true}", **kw)
        class_scatter(fig_a, x[p1[a]], y_a[p1[a]], rescale=False)
        if n_matched == 0:
            figures.append([fig_a, None, None, None])
            continue
        fig_b = bk.figure(
            title=(
                f"{len(bs)} matched classes: "
                + ", ".join(map(str, bs))
                + f"; n = {n_matched}"
            ),
            **kw,
        )
        class_scatter(fig_b, x[p2[a]], y_b[p2[a]], rescale=False)
        fig_match = bk.figure(title=f"Intersection; n = {n_inter}", **kw)
        class_scatter(
            fig_match, x[p1[a] & p2[a]], y_b[p1[a] & p2[a]], rescale=False
        )
        y_diff = p3[a] + 2 * p4[a]
        fig_diff = bk.figure(
            title=(
                f"Symmetric difference; n = {n_miss + n_exc}\n"
                f"Misses (red) = {n_miss}; excess (blue) = {n_exc}"
            ),
            **kw,
        )
        class_scatter(
            fig_diff,
            x[y_diff > 0],
            y_diff[y_diff > 0],
            palette=["#ff0000", "#0000ff"],
            rescale=False,
        )
        make_same_xy_range(fig_a, fig_b, fig_match, fig_diff)
        figures.append([fig_a, fig_b, fig_match, fig_diff])

    return bkl.grid(figures)  # type: ignore


def make_same_xy_range(*args: bk.figure) -> None:
    """Makes sure all figures share the same `x_range` and `y_range`"""
    for f in args[1:]:
        f.x_range, f.y_range = args[0].x_range, args[0].y_range
