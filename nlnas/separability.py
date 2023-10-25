"""Everything related to separability of point clouds"""


from itertools import combinations
import random
from math import sqrt
from typing import Tuple

import numpy as np
import torch
from sklearn.svm import SVC
from torch import Tensor
from torch.nn.functional import one_hot, pdist
from tqdm import tqdm


def _var(x: Tensor) -> Tensor:
    """Variance along dimension 0"""
    return torch.mean(x**2, dim=0) - torch.mean(x, dim=0) ** 2


def gdv(x: Tensor, y: Tensor) -> Tensor:
    """
    Generalized Discrimination Value of

        Achim Schilling, Andreas Maier, Richard Gerum, Claus Metzner, Patrick
        Krauss, Quantifying the separability of data classes in neural
        networks, Neural Networks, Volume 139, 2021, Pages 278-293, ISSN
        0893-6080, https://doi.org/10.1016/j.neunet.2021.03.035.
        (https://www.sciencedirect.com/science/article/pii/S0893608021001234)

    This method is differentiable. The complexity is quadratic in the number of
    classes and samples.

    Args:
        x (Tensor): A `(N, ...)` tensor, where `N` is the number of samples. It
            will automatically be flattened to a 2 dimensional tensor.
        y (Tensor): A `(N,)` tensor.

    Returns:
        A scalar tensor
    """
    s = x.flatten(1)
    s = 0.5 * (s - s.mean(dim=0)) / (_var(s).sqrt() + 1e-5)
    n_classes = len(y.unique())
    a, b = [], []
    for i in range(n_classes):
        u = s[y == i]
        if len(u) >= 2:
            a.append(pdist(u).mean())
    for i, j in combinations(range(n_classes), 2):
        u, v = s[y == i], s[y == j]
        if len(u) >= 1 and len(v) >= 1:
            b.append(torch.cdist(u, v).mean())
    d = sqrt(s.shape[-1])
    return (torch.stack(a).mean() - 2 * torch.stack(b).mean()) / d
    # GAUSSIAN APPROXIMATION
    # for i in range(n_classes):
    #     u = s[y == i] / 2
    #     a.append(2 * _var(u).sum())
    # for i, j in combinations(range(n_classes), 2):
    #     u, v = s[y == i] / 2, s[y == j] / 2
    #     d = (
    #         torch.linalg.norm(u.mean(dim=0) - v.mean(dim=0))
    #         + _var(u).sum()
    #         + _var(v).sum()
    #     )
    #     b.append(d)
    # return (
    #     (torch.stack(a).mean() - 2 * torch.stack(b).mean())
    #     / sqrt(n_classes)
    # )


def label_variation(
    x: Tensor,
    y: Tensor,
    k: int,
    n_classes: int = -1,
    sigma: float = 1.0,
) -> Tensor:
    """
    Label variation metric of

        Lassance, C.; Gripon, V.; Ortega, A. Representing Deep Neural Networks
        Latent Space Geometries with Graphs. Algorithms 2021, 14, 39.
        https://doi.org/10.3390/a14020039

    with a few tweaks.

    TODO: list the tweaks =]

    This method is differentiable, but a call to
    [`torch.cdist`](https://pytorch.org/docs/stable/generated/torch.cdist.html)
    is needed which makes its cost quadratic in `N`, where `N` is the the
    number of samples, aka the length of `x`.

    Args:
        x (Tensor): Sample tensor with shape `(N, d)`, where `d` the latent
            dimension
        y (Tensor): Label vector with shape `(N,)`
        k (int): Number of nearest neighbors to consider when thresholding the
            distance matrix of `x`
        n_classes (int): Leave it to `-1` to automatically infer the number of
            classes
        sigma (float, optional): RBF parameter

    Returns:
        A scalar tensor
    """
    # p = torch.exp(pdist(x) / (2 * sigma**2))
    # c = torch.tensor(list(combinations(range(len(x)), 2)))
    # dm = torch.sparse_coo_tensor(c.T, p, size=(len(x), len(x))).to_dense()
    # dm = dm + dm.T
    s = x.flatten(1)
    s = (s - s.mean(dim=0)) / (_var(s).sqrt() + 1e-5)
    dm = torch.cdist(s, s) / sqrt(s.shape[-1])
    dm = torch.exp(-dm / (2 * sigma**2))
    t0 = dm.topk(k + 1, dim=0, largest=False).values[-1]
    t1 = dm.topk(k + 1, dim=1, largest=False).values[:, [-1]]
    m0, m1 = dm * (dm <= t0), dm * (dm <= t1)
    a = torch.maximum(m0, m1)
    y_oh = one_hot(y.long(), n_classes).float()
    return torch.trace(y_oh.T @ a @ y_oh) / (
        y_oh.shape[0] * y_oh.shape[-1]  # * k
    )


def pairwise_gdv(
    x: np.ndarray | Tensor,
    y: np.ndarray | Tensor,
    max_class_pairs: int = 100,
) -> Tuple[list[dict], float]:
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

        and a float which is the mean of all `value`'s
    """
    x = Tensor(x) if isinstance(x, np.ndarray) else x
    y = Tensor(y) if isinstance(y, np.ndarray) else y
    n_classes = len(np.unique(y))
    class_idx_pairs = list(combinations(range(n_classes), 2))
    if (n_classes * (n_classes - 1) / 2) > max_class_pairs:
        class_idx_pairs = random.sample(class_idx_pairs, max_class_pairs)
    result = []
    progress = tqdm(class_idx_pairs, desc="Computing GDVs", leave=False)
    for i, j in progress:
        progress.set_postfix({"i": i, "j": j})
        yij = (y == i) + (y == j)
        a, b = x[yij], y[yij] == i
        result.append({"idx": (i, j), "value": float(gdv(a, b))})
    return result, float(np.mean([d["value"] for d in result]))  # type: ignore


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
    class_idx_pairs = list(combinations(range(n_classes), 2))
    if (n_classes * (n_classes - 1) / 2) > max_class_pairs:
        class_idx_pairs = random.sample(class_idx_pairs, max_class_pairs)
    results = []
    for i, j in class_idx_pairs:
        yij = (y == i) + (y == j)
        a, b = x[yij], y[yij] == i
        svc = SVC(**kwargs).fit(a, b)
        results.append({"idx": (i, j), "score": svc.score(a, b)})
    return results
