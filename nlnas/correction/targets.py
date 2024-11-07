"""
Module dedicated to fining LCC targets.

LCC is all about detecting and correcting misclustered samples. Detection occurs
at the clustering stage, see `nlnas.correction.clustering` and specifically
`nlnas.correction.clustering.otm_matching_predicates`.

Then comes the question of correction. Misclustered samples are pulled towards a
_target_ that is itself correctly clustered (and in the same class). This module
is dedicated to finding and/or choosing these targets.
"""

from collections import defaultdict
from math import sqrt

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import Tensor
from torch.utils.data import DataLoader

from ..utils import TqdmStyle, make_tqdm, to_int_array, to_int_tensor
from .clustering import otm_matching_predicates
from .utils import Matching, to_int_matching


def _mc_cc_predicates(
    y_true: ArrayLike,
    y_clst: ArrayLike,
    matching: Matching,
    n_true_classes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns two boolean arrays (also called predicates) `p_mc` and `p_cc` (in
    this order), both of shape `(n_true_classes, N)`, where:
    - `p_mc[i_true, j]` is `True` if the $j$-th sample is in true class `i_true`
      and misclustered (i.e. not in any cluster matched with true class
      `i_true`);
    - `p_cc[i_true, j]` is `True` if the $j$-th sample is in true class `i_true`
      and correctly clustered (i.e. in a cluster matched with true class
      `i_true`).

    Note:
        `p_mc != ~p_cc` in general ;)

    Args:
        y_true (ArrayLike): A `(N,)` integer array.
        y_clst (ArrayLike): A `(N,)` integer array.
        matching (Matching):
        n_true_classes (int | None, optional): Number of true classes. Useful if
            `y_true` is a slice of the real true label vector and does not
            contain all the possible true classes of the dataset at hand.  If
            `None`, then `y_true` is assumed to contain all classes, and so
            `n_true_classes` defaults to `y_true.max() + 1`.
    """
    y_true, y_clst = to_int_array(y_true), to_int_array(y_clst)
    p1, p2, p_mc, _ = otm_matching_predicates(
        y_true, y_clst, matching, c_a=n_true_classes or int(y_true.max() + 1)
    )
    return p_mc, p1 & p2


def lcc_loss(
    z: Tensor,
    y_true: ArrayLike,
    y_clst: ArrayLike,
    matching: Matching,
    targets: dict[int, Tensor],
    n_true_classes: int | None = None,
) -> Tensor:
    """
    Derives the clustering correction loss from a tensor of latent
    representation `z` and dict of targets (see `nlnas.correction.lcc_targets`).

    First, recall that the values of `target` (as produced
    `nlnas.correction.lcc_targets`) are `(k, d)` tensors, for some length `k`.

    Let's say `a` is a missclusterd latent sample (a.k.a. a row of `z`) in true
    class `i_true`, and that `(b_1, ..., b_k)` are the rows of
    `targets[i_true]`. Then `a` contributes a term to the LCC loss equal to the
    distance between `a` and the closest `b_j`, divided by $\\\\sqrt{d}$.

    It is possible that `i_true` is not in the keys of `targets`, in which case
    the contribution of `a` to the LCC loss is zero. In particular, if `targets`
    is empty, then the LCC loss is zero.

    Usage:

        ```python
        targets = lcc_targets(
            dl,
            y_true,
            y_clst,
            matching,
            n_true_classes=n_true_classes,
        )
        loss = lcc_loss(
            z,
            y_true,
            y_clst,
            matching,
            targets,
            n_true_classes=n_true_classes,
        )
        ```

    Args:
        z (Tensor): The tensor of latent representations. *Do not* mask it
            before passing it to this method.  The correctly samples and the
            missclustered samples are automatically separated.
        y_true (ArrayLike): A `(N,)` integer array of true
            labels.
        y_clst (ArrayLike): A `(N,)` integer array of the
            cluster labels.
        matching (Matching): As produced by
            `nlnas.correction.class_otm_matching`.
        targets (dict[int, Tensor]): As produced by
            `nlnas.correction.lcc_targets`.
        n_true_classes (int | None, optional): Number of true classes. Useful if
            `y_true` is a slice of the actual true label vector and does not
            contain all the possible true classes.
    """
    z = z.flatten(1)
    if not targets:
        # ↓ actually need grad?
        return torch.tensor(0.0, requires_grad=True).to(z.device)
    y_true = to_int_tensor(y_true)
    p_mc, _ = _mc_cc_predicates(
        y_true, y_clst, matching, n_true_classes=n_true_classes
    )
    sqrt_d, losses = sqrt(z.shape[-1]), []
    for i_true, p_mc_i_true in enumerate(p_mc):
        if not (i_true in targets and len(targets[i_true]) > 0):  # no targets
            continue
        if not p_mc_i_true.any():  # every sample is correctly clustered
            continue
        d = torch.cdist(z[p_mc_i_true], targets[i_true]) / sqrt_d
        losses.append(d.min(dim=-1).values)
    if not losses:
        return torch.tensor(0.0, requires_grad=True).to(z.device)
    return torch.concat(losses).mean()


def lcc_targets(
    dl: DataLoader,
    y_true: ArrayLike,
    y_clst: ArrayLike,
    matching: Matching,
    n_true_classes: int | None = None,
    ccspc: int = 1,
    tqdm_style: TqdmStyle = None,
) -> dict[int, Tensor]:
    """
    Provides the correction targets for misclustered samples in `z`. In more
    details, this method returns a dict where:
    - the keys are *among* true classes (unique values of `y_true`) and in fact
      are the same keys as `knn_indices`; let's say that `i_true` is a key that
      owns `k` clusters;
    - the associated value a `(n, d)` tensor, where `d` is the latent dimension,
      whose rows are among correctly clustered samples in true class `i_true`.
      If `ccspc` is $1$, then `n` is the number of clusters matched with
      `i_true`, say `k`. Otherwise, `n <= k * ccspc`.

    Under the hood, this method first choose the samples by their index based on
    the "correctly clustered" predicate of `_mc_cc_predicates`. Then, the whole
    dataset is iterated to collect the actual samples.

    Args:
        dl (DataLoader): A dataloader over a tensor dataset.
        y_true (ArrayLike): A `(N,)` integer array.
        y_clst (ArrayLike): A `(N,)` integer array.
        matching (Matching): Produced by
            `nlnas.correction.class_otm_matching`.
        knn_indices (dict[int, tuple[Any, Tensor]]): As produced by
            `lcc_knn_indices`
        n_true_classes (int | None, optional): Number of true classes. Useful if
            `y_true` is a slice of the real true label vector and does not
            contain all the possible true classes of the dataset at hand. If
            `None`, then `y_true` is assumed to contain all classes, and so
            `n_true_classes` defaults to `y_true.max() + 1`.
        ccspc (int, optional): Maximum number of correctly clustered samples to
            collect per cluster.
        tqdm_style (TqdmStyle, optional):
    """
    matching = to_int_matching(matching)
    _, p_cc = _mc_cc_predicates(y_true, y_clst, matching, n_true_classes)
    indices: dict[int, list[int]] = defaultdict(list)
    for i_true, p_cc_i_true in enumerate(p_cc):
        for j_clst in matching[i_true]:
            p = p_cc_i_true & (y_clst == j_clst)
            ccids = list(np.unique(np.random.choice(np.where(p)[0], ccspc)))
            indices[i_true] += ccids
    n_seen, n_todo = 0, sum(len(v) for v in indices.values())
    result: dict[int, list[Tensor]] = defaultdict(list)
    progress = make_tqdm(tqdm_style)(
        dl, "Finding correction targets ({ccspc=})"
    )
    for batch in progress:
        for i_true, idxs in indices.items():
            lst = [idx for idx in idxs if n_seen <= idx < n_seen + len(batch)]
            for idx in lst:
                result[i_true].append(batch[idx - n_seen])
                n_todo -= 1
        if n_todo <= 0:
            return {k: torch.stack(v).flatten(1) for k, v in result.items()}
        n_seen += len(batch)
    raise RuntimeError("Some correction targets could not be found")
