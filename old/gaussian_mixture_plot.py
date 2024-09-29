import bokeh.plotting as bk
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture


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
