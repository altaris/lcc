"""Plotting utilities"""


import bokeh.plotting as bk
import bokeh.palettes as plt
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

BK_PALETTE_FUNCTIONS = {
    "cividis": plt.cividis,
    "gray": plt.gray,
    "grey": plt.grey,
    "inferno": plt.inferno,
    "magma": plt.magma,
    "viridis": plt.viridis,
}


def class_scatter(
    plot: bk.figure,
    x: np.ndarray,
    y: np.ndarray,
    palette: plt.Palette | list[str] | str | None = None,
) -> None:
    """
    Scatter plot where each class has a different color

    Args:
        plot (bk.figure):
        x (np.ndarray): A `(N, 2)` array
        y (np.ndarray): A `(N,)` int array. Each unique value corresponds to a
            class
        palette (plt.Palette | list[str] | str | None, optional): Either a
            palette object, a list of HTML colors (at least as many as the
            number of classes), or a name in
            `nlnas.plotting.BK_PALETTE_FUNCTIONS`.

    Raises:
        ValueError: _description_
    """
    n_classes = len(np.unique(y))
    if palette is None:
        palette = plt.gray(n_classes)
    if isinstance(palette, str):
        if palette not in BK_PALETTE_FUNCTIONS:
            raise ValueError(f"Unknown palette '{palette}'")
        palette = BK_PALETTE_FUNCTIONS[palette](n_classes)
    for j in range(n_classes):
        a = x[y == j]
        plot.scatter(
            a[:, 0],
            a[:, 1],
            color=palette[j],
            line_width=0,
            size=3,
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

    Example usage:

    ```py
    import bokeh.plotting as bk
    plot = bk.figure()
    gaussian_mixture_plot(plot, gm)
    bk.show(plot)
    ```

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
