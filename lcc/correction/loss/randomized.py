"""
Randomized LCC loss, which is a cheaper version of
`lcc.correction.ExactLCCLoss`.
"""

from collections import defaultdict
from math import sqrt
from typing import Any

import numpy as np
import torch
from lightning_fabric import Fabric
from loguru import logger as logging
from numpy.typing import ArrayLike
from pytorch_lightning.strategies import Strategy
from safetensors import torch as st
from torch import Tensor
from torch.utils.data import DataLoader

from ...utils import TqdmStyle, make_tqdm, to_int_array, to_int_tensor
from ..clustering import otm_matching_predicates
from ..utils import Matching, to_int_matching
from .base import LCCLoss


class RandomizedLCCLoss(LCCLoss):
    """
    A LCC loss function that pulls misclustered samples towards a CC sample in
    the same class.

    In principle, this implies some sort of exhaustive search since a MC sample
    has to be compared to *every* CC sample in the same class. This is what
    `lcc.correction.ExactLCCLoss` does. Here, to save on compute and time, only
    a few CC samples are randomly selected in each cluster and used as potential
    targets.
    """

    ccspc: int
    n_classes: int
    targets: dict[int, Tensor] = {}
    tqdm_style: TqdmStyle
    matching: Matching

    def __call__(
        self, z: Tensor, y_true: ArrayLike, y_clst: ArrayLike
    ) -> Tensor:
        """
        Derives the clustering correction loss from a tensor of latent
        representation `z` and dict of targets (see
        `lcc.correction.lcc_targets`).

        First, recall that the values of `target` (as produced
        `lcc.correction.lcc_targets`) are `(k, d)` tensors, for some length
        `k`.

        Let's say `a` is a misclustered latent sample (a.k.a. a row of `z`) in
        true class `i_true`, and that `(b_1, ..., b_k)` are the rows of
        `targets[i_true]`. Then `a` contributes a term to the LCC loss equal to
        the distance between `a` and the closest `b_j`, divided by
        $\\\\sqrt{d}$.

        It is possible that `i_true` is not in the keys of `targets`, in which
        case the contribution of `a` to the LCC loss is zero. In particular, if
        `targets` is empty, then the LCC loss is zero.

        Args:
            z (Tensor): The tensor of latent representations. *Do not* mask it
                before passing it to this method.  The correctly samples and the
                missclustered samples are automatically separated.
            y_true (ArrayLike): A `(N,)` integer array of true labels.
            y_clst (ArrayLike): A `(N,)` integer array of the cluster labels.
        """
        if not self.targets:
            # ↓ actually need grad?
            return torch.tensor(0.0, requires_grad=True).to(z.device)
        z, y_true = z.flatten(1), to_int_tensor(y_true)
        p_mc, _ = _mc_cc_predicates(
            y_true, y_clst, self.matching, n_classes=self.n_classes
        )
        sqrt_d, losses = sqrt(z.shape[-1]), []
        for i_true, p_mc_i_true in enumerate(p_mc):
            if not (
                i_true in self.targets and len(self.targets[i_true]) > 0
            ):  # no targets in this true class
                continue
            if not p_mc_i_true.any():  # every sample is correctly clustered
                continue
            d = (
                torch.cdist(z[p_mc_i_true], self.targets[i_true].to(z.device))
                / sqrt_d
            )
            losses.append(d.min(dim=-1).values)
        if not losses:
            return torch.tensor(0.0, requires_grad=True).to(z.device)
        return torch.concat(losses).mean()

    def __init__(
        self,
        n_classes: int,
        ccspc: int = 1,
        tqdm_style: TqdmStyle = None,
        strategy: Strategy | Fabric | None = None,
    ) -> None:
        super().__init__(strategy=strategy)
        self.n_classes, self.ccspc = n_classes, ccspc
        self.tqdm_style = tqdm_style

    def sync(self, **kwargs: Any) -> None:
        if self.strategy is None:
            return
        path = self._get_tmp_dir()
        gr = self.strategy.global_rank
        st.save_file(
            {str(k): v for k, v in self.targets.items()},
            path / f"targets.{gr}",
        )
        self.strategy.barrier()
        for r in range(self.strategy.world_size):
            if r == self.strategy.global_rank:
                continue  # data from this rank is already in self.targets
            data = st.load_file(path / f"targets.{r}")
            self.targets.update({int(k): v for k, v in data.items()})
        return super().sync(**kwargs)

    def update(
        self,
        dl: DataLoader,
        y_true: ArrayLike,
        y_clst: ArrayLike,
        matching: Matching,
    ) -> None:
        """
        This method updates the `targets` attribute of this instance. It is a
        dict containing the following:
        - the keys are *among* true classes (unique values of `y_true`); let's
          say that `i_true` is a key that owns `k` clusters;
        - the associated value a `(n, d)` tensor, where `d` is the latent
          dimension, whose rows are among correctly clustered samples in true
          class `i_true`.  If `ccspc` is $1$, then `n` is the number of clusters
          matched with `i_true`, say `k`. Otherwise, `n <= k * ccspc`.

        Under the hood, this method first choose the samples by their index
        based on the "correctly clustered" predicate of `_mc_cc_predicates`.
        Then, the whole dataset is iterated to collect the actual samples.

        Args:
            dl (DataLoader): An unsharded dataloader over a tensor dataset.
            y_true (ArrayLike): A `(N,)` integer array.
            y_clst (ArrayLike): A `(N,)` integer array.
            matching (Matching): Produced by
               `lcc.correction.class_otm_matching`.
        """
        self.matching = to_int_matching(matching)
        _, p_cc = _mc_cc_predicates(
            y_true, y_clst, self.matching, self.n_classes
        )
        # i_true (assigned to this rank) -> some indices of CC samples
        indices: dict[int, set[int]] = {
            i_true: set() for i_true in self._distribute_labels(y_true)
        }
        for i_true in indices:
            for j_clst in self.matching[i_true]:
                # p: (N,) CC in i_true and j_clst
                p = p_cc[i_true] & (y_clst == j_clst)
                s = np.random.choice(np.where(p)[0], self.ccspc)
                indices[i_true].update(s)
        n_seen, n_todo = 0, sum(len(v) for v in indices.values())
        # i_true (assigned to this rank) -> some CC samples
        result: dict[int, list[Tensor]] = defaultdict(list)
        tqdm = make_tqdm(self.tqdm_style)
        progress = tqdm(dl, f"Finding correction targets (ccspc={self.ccspc})")
        for z, *_ in progress:
            for i_true, idxs in indices.items():
                lst = [idx for idx in idxs if n_seen <= idx < n_seen + len(z)]
                for idx in lst:
                    result[i_true].append(z[idx - n_seen])
                    n_todo -= 1
            if n_todo <= 0:
                break
            n_seen += len(z)
        if n_todo > 0:
            logging.warning(
                "Some correction targets could not be found "
                "(n_seen={}, n_todo={})",
                n_seen,
                n_todo,
            )
        self.targets = {
            k: torch.stack(v).flatten(1)
            for k, v in result.items()
            if v  # should already be is non-empty but just to make sure...
        }


def _mc_cc_predicates(
    y_true: ArrayLike,
    y_clst: ArrayLike,
    matching: Matching,
    n_classes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns two boolean arrays (also called predicates) `p_mc` and `p_cc` (in
    this order), both of shape `(n_classes, N)`, where:
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
        n_classes (int | None, optional): Number of true classes. Useful if
            `y_true` is a slice of the real true label vector and does not
            contain all the possible true classes of the dataset at hand.  If
            `None`, then `y_true` is assumed to contain all classes, and so
            `n_classes` defaults to `y_true.max() + 1`.
    """
    y_true, y_clst = to_int_array(y_true), to_int_array(y_clst)
    p1, p2, p_mc, _ = otm_matching_predicates(
        y_true, y_clst, matching, c_a=n_classes or int(y_true.max() + 1)
    )
    return p_mc, p1 & p2
