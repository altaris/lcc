"""Everything related to separability of point clouds"""


import random
from math import sqrt

import numpy as np
import torch
from sklearn.svm import SVC
from torch import Tensor


def pairwise_gdv(
    x: np.ndarray | Tensor,
    y: np.ndarray | Tensor,
    max_class_pairs: int = 100,
) -> list[dict]:
    """
    Pairwise GDVs, see `nlnas.separability.gdv`.

    Args:
        x (np.ndarray | Tensor): A `(N, ...)` tensor or array, where `N` is the
            number of samples. It will automatically be flattened to a 2
            dimensional tensor.
        y (np.ndarray | Tensor): A `(N,)` tensor or array.
        max_class_pairs (int): The number of calls to `nlnas.separability.gdv`
            this method needs to make is quadratic in the number of classes. If
            there are many classes, this can be prohibitively expensive. So, if
            the number of class pairs exceeds `max_class_pairs`, only
            `max_class_pairs` pairs chosen at random are considered

    Returns:
        A dict that looks like this:

            [
                {
                    "idx": <a pair of int>,
                    "value": <a float>,
                },
                ...
            ]
    """
    x = Tensor(x) if isinstance(x, np.ndarray) else x
    y = Tensor(y) if isinstance(y, np.ndarray) else y
    n_classes = len(np.unique(y))
    class_idx_pairs = [
        (i, j) for i in range(n_classes) for j in range(i + 1, n_classes)
    ]
    if (n_classes * (n_classes - 1) / 2) > max_class_pairs:
        class_idx_pairs = random.sample(class_idx_pairs, max_class_pairs)
    results = []
    for i, j in class_idx_pairs:
        yij = (y == i) + (y == j)
        a, b = x[yij], y[yij] == i
        results.append({"idx": (i, j), "value": float(gdv(a, b))})
    return results


def pairwise_svc_scores(
    x: np.ndarray | Tensor,
    y: np.ndarray | Tensor,
    max_class_pairs: int = 100,
    **kwargs,
) -> list[dict]:
    """
    Args:
        x (Tensor): A `(N, ...)` tensor, where `N` is the number of samples. It
            will automatically be flattened to a 2 dimensional tensor.
        y (Tensor): A `(N,)` tensor.
        max_class_pairs (int): The number of SVCs this method needs to fit is
            quadratic in the number of classes. If there are many classes, this
            can be prohibitively expensive. So, if the number of class pairs
            exceeds `max_class_pairs`, only `max_class_pairs` pairs chosen at
            random are considered
        kwargs: Passed to [`SVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

    Returns:
        A dict that looks like this:

            [
                {
                    "idx": <a pair of int>,
                    "score": <a float>,
                },
                ...
            ]
    """
    x = x.numpy() if isinstance(x, Tensor) else x
    y = y.numpy() if isinstance(y, Tensor) else y
    x = x.reshape(len(x), -1)
    n_classes = len(np.unique(y))
    class_idx_pairs = [
        (i, j) for i in range(n_classes) for j in range(i + 1, n_classes)
    ]
    if (n_classes * (n_classes - 1) / 2) > max_class_pairs:
        class_idx_pairs = random.sample(class_idx_pairs, max_class_pairs)
    results = []
    for i, j in class_idx_pairs:
        yij = (y == i) + (y == j)
        a, b = x[yij], y[yij] == i
        svc = SVC(**kwargs).fit(a, b)
        results.append({"idx": (i, j), "score": svc.score(a, b)})
    return results


def gdv(x: Tensor, y: Tensor) -> Tensor:
    """
    Gaussian approximation of the Generalized Discrimination Value of

        Achim Schilling, Andreas Maier, Richard Gerum, Claus Metzner, Patrick
        Krauss, Quantifying the separability of data classes in neural
        networks, Neural Networks, Volume 139, 2021, Pages 278-293, ISSN
        0893-6080, https://doi.org/10.1016/j.neunet.2021.03.035.
        (https://www.sciencedirect.com/science/article/pii/S0893608021001234)

    This method is differentiable. The complexity is quadratic in the number of
    classes.

    Args:
        x (Tensor): A `(N, ...)` tensor, where `N` is the number of samples. It
            will automatically be flattened to a 2 dimensional tensor.
        y (Tensor): A `(N,)` tensor.
    """

    def _var(c: Tensor) -> Tensor:
        """Variance along dimension 0"""
        return torch.mean(c**2, dim=0) - torch.mean(c, dim=0) ** 2

    a, b, n_classes = [], [], len(y.unique())
    s = x.flatten(1)
    s = 0.5 * (s - s.mean(dim=0)) / (_var(s).sqrt() + 1e-5)
    for i in range(n_classes):
        u = s[y == i] / 2
        a.append(2 * _var(u).sum())
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            u, v = s[y == i] / 2, s[y == j] / 2
            d = (
                torch.linalg.norm(u.mean(dim=0) - v.mean(dim=0))
                + _var(u).sum()
                + _var(v).sum()
            )
            b.append(d)
    return sqrt(n_classes) * (
        torch.stack(a).mean() - 2 * torch.stack(b).mean()
    )
