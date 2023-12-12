"""Everything related to separability of point clouds"""


from itertools import combinations
from math import sqrt
from typing import Iterable

import torch
from torch import Tensor
from torch.nn.functional import one_hot, pdist

SQRT_2 = 1.4142135623730951


def _img_orthn_basis(x: Tensor) -> Tensor:
    """
    Given a `(N, d)` design matrix, where `N` is the batch size and `d` is the
    latent dimension returns a `(r, d)` matrix whose rows form an orthonormal
    basis of the subspace of $\\mathbb{R}^d$ spanned by the dataset vectors.

    Args:
        x (Tensor): A `(N, d)` design matrix, where `N` is the batch size and
            `d` is the latent dimension

    Returns:
        A `(r, d)` tensor that is differentiable in `x`, where `r` is the rank
        of `x.T`
    """
    # x is a "row-wise" design matrix (one where the feature vectors are the
    # rows), but torch.linalg.svd expects a "column-wise" design matrix. u is
    # also column-wise
    u, s, _ = torch.linalg.svd(x.T)
    return u.T[: len(s)][s > 1e-5]


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


def ggd(
    a: Tensor,
    b: Tensor,
    orthonormal: bool = False,
) -> Tensor:
    """
    Grassmanian geodesic distance between the subspaces spanned by two design
    matrices `a` and `b`. These don't need to be orthonormal or have the same
    rank. See table 3 of

        Ye, Ke, and Lek-Heng Lim. "Schubert varieties and distances between
        subspaces of different dimensions." SIAM Journal on Matrix Analysis and
        Applications 37.3 (2016): 1176-1197.

    Args:
        a (Tensor): A `(N, d)` design matrix, where `N` is the batch size and
            `d` is the latent dimension
        b (Tensor): A `(N, d)` design matrix
        orthonormal (bool, optional): Wether the rows of `a` form an
            orthonormal family (and same for `b`).

    Returns:
        A scalar tensor that is differentiable in `a` and `b`
    """
    w_a, w_b = (
        (a, b) if orthonormal else (_img_orthn_basis(a), _img_orthn_basis(b))
    )  # both (?, d)
    cos_theta = torch.linalg.svdvals(w_a @ w_b.T)
    cos_theta = torch.min(cos_theta, torch.ones_like(cos_theta))
    theta = torch.acos(cos_theta)
    r_corr = abs(w_a.shape[0] - w_b.shape[0]) * torch.pi**2 / 4
    return torch.sqrt(r_corr + (theta**2).sum())


def mean_ggd(
    x: Tensor,
    y: Tensor,
    classes: Iterable[int] | None = None,
) -> Tensor:
    """
    Mean Grassmanian distance between all classes.

    Warning:
        If $k$ is the number of classes (distinct values of `y` if `n_classes`
        is not specified), then this methods makes $k$ calls to
        `torch.linalg.svd` which can be expensive.

    Args:
        x (Tensor): A `(N, d)` design matrix, where `N` is the batch size and
            `d` is the latent dimension
        y (Tensor): A `(N,)` label tensor
        classes (Iterable[int], optional): Leave to `None` to automatically
            find the class labels from `y`.

    Returns:
        A scalar tensor that is differentiable in `x`
    """
    classes = classes or list(y.unique(sorted=False))
    assert classes is not None  # for typechecking
    bs = {i: _img_orthn_basis(x[y == i]) for i in classes}
    ds = [
        ggd(bs[i], bs[j], orthonormal=True)
        for i, j in combinations(classes, 2)
    ]
    return torch.stack(ds).mean()


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
