"""Exact LCC loss based on KNN correctly clustered samples."""

from math import sqrt

import faiss
import faiss.contrib.torch_utils
import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import Tensor
from torch.utils.data import DataLoader

from ...utils import (
    TqdmStyle,
    make_tqdm,
    to_array,
    to_int_array,
    to_int_tensor,
)
from ..clustering import otm_matching_predicates
from ..utils import Matching, to_int_matching
from .base import LCCLoss


class ExactLCCLoss(LCCLoss):
    """
    LCC loss that corrects missclustered samples using their CC KNNs. This
    differs from `nlnas.correction.loss.RandomizedLCCLoss`, where targets are
    chosen randomly.
    """

    k: int
    n_classes: int
    tqdm_style: TqdmStyle
    matching: dict[int, set[int]]

    data: dict[int, tuple[faiss.IndexHNSWFlat, Tensor]] = {}
    # i_clst -> (knn index, tensor of all CC samples in this clst)

    def __call__(
        self, z: Tensor, y_true: ArrayLike, y_clst: ArrayLike
    ) -> Tensor:
        _, _, p_mc, _ = otm_matching_predicates(
            y_true, y_clst, self.matching, c_a=self.n_classes
        )  # p_mc: (n_classes, len(z))
        terms = []
        for i_true in np.unique(to_int_array(y_true)):
            if not p_mc[i_true].any():
                # No MC samples in this class and batch
                continue
            u = z[p_mc[i_true]]  # MC samples in class i_true
            d = []  # Distances of MC samples to candidate targets
            for i_clst in self.matching[i_true]:
                if i_clst not in self.data:
                    continue
                knn, cc = self.data[i_clst]
                # Find a candidate target in i_clst for each sample in u
                _, j = knn.search(to_array(u).astype(np.float32), self.k)
                j = to_int_tensor(j)  # (len(u), k)
                t = cc[j].mean(dim=1).to(u.device)  # (n, n_features)
                # Save distance to these candidate targets
                d.append(
                    torch.norm(u - t, dim=1) / sqrt(u.shape[-1])  # (len(u),)
                )
            if d:
                terms.append(
                    torch.stack(d)  # (?, len(u))
                    .min(dim=0)
                    .values  # (len(u),)
                )
        if not terms:
            return torch.tensor(0.0, requires_grad=True).to(z.device)
        return torch.cat(terms).mean()

    def __init__(
        self,
        n_classes: int,
        k: int = 5,
        tqdm_style: TqdmStyle = None,
    ) -> None:
        super().__init__()
        self.k, self.n_classes = k, n_classes
        self.tqdm_style = tqdm_style

    def update(
        self,
        dl: DataLoader,
        y_true: ArrayLike,
        y_clst: ArrayLike,
        matching: Matching,
    ) -> None:
        self.matching = to_int_matching(matching)
        y_clst, n_clst = to_int_array(y_clst), len(np.unique(y_clst))
        n_features = next(iter(dl)).flatten(1).shape[-1]
        p1, p2, _, _ = otm_matching_predicates(
            y_true, y_clst, self.matching, c_a=self.n_classes
        )
        p_cc = (p1 & p2).sum(axis=0).astype(bool)  # (n_samples,)
        data: dict[int, tuple[faiss.IndexHNSWFlat, list[Tensor]]] = {
            i_clst: (faiss.IndexHNSWFlat(n_features, self.k), [])
            for i_clst in range(n_clst)
        }
        tqdm, n_seen = make_tqdm(self.tqdm_style), 0
        for batch in tqdm(dl, f"Building {n_clst} KNN indices"):
            z = batch.flatten(1)  # (b, n_feat.)
            _y_clst = y_clst[n_seen : n_seen + len(batch)]
            _p_cc = p_cc[n_seen : n_seen + len(batch)]  # (b,)
            for i_clst in np.unique(_y_clst):
                _p_cc_i_clst = _p_cc & (_y_clst == i_clst)
                if not _p_cc_i_clst.any():
                    # No CC sample in cluster i_clst in this batch
                    continue
                data[i_clst][0].add(
                    to_array(z[_p_cc_i_clst]).astype(np.float32)
                )
                data[i_clst][1].append(z[_p_cc_i_clst])
            n_seen += len(batch)
        self.data = {
            i_clst: (idx, torch.cat(smpls))
            for i_clst, (idx, smpls) in data.items()
            if smpls
        }
