"""Exact LCC loss based on KNN correctly clustered samples."""

from math import sqrt
from typing import Any

import faiss
import numpy as np
import torch
from lightning_fabric import Fabric
from numpy.typing import ArrayLike
from pytorch_lightning.strategies import Strategy
from safetensors import torch as st
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
    """LCC loss that corrects missclustered samples using their CC KNNs"""

    k: int
    n_classes: int
    tqdm_style: TqdmStyle
    matching: dict[int, set[int]]

    # ↓ i_clst -> (knn idx, tensor of all CC samples in that clst)
    data: dict[int, tuple[faiss.IndexHNSWFlat, Tensor]] = {}

    def __call__(
        self, z: Tensor, y_true: ArrayLike, y_clst: ArrayLike
    ) -> Tensor:
        _, _, p_mc, _ = otm_matching_predicates(
            y_true, y_clst, self.matching, c_a=self.n_classes
        )  # p_mc: (n_classes, len(z))
        z = z.flatten(1)
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
        strategy: Strategy | Fabric | None = None,
    ) -> None:
        super().__init__(strategy=strategy)
        self.k, self.n_classes = k, n_classes
        self.tqdm_style = tqdm_style

    def sync(self, **kwargs: Any) -> None:
        """
        Remember that every rank has its own subset of cluster to manage. Before
        sync every rank's `self.data` only contains data pertaining to this
        rank's clusters.

        This method works in two steps. First, every rank writes its data to
        some temporary directory. Then, every rank loads data from that
        directory.

        EZPZ
        """
        if self.strategy is None:
            return
        path = self._get_tmp_dir()
        gr = self.strategy.global_rank
        st.save_file(
            {str(i_clst): cc for i_clst, (_, cc) in self.data.items()},
            path / f"cc.{gr}",
        )
        for i_clst, (idx, _) in self.data.items():
            faiss.write_index(idx, str(path / f"knn.{i_clst}.{gr}"))
        self.strategy.barrier()
        for r in range(self.strategy.world_size):
            if r == self.strategy.global_rank:
                continue  # data from this rank is already in self.data
            ccs = st.load_file(path / f"cc.{r}")
            for i_clst, cc in ccs.items():  # type: ignore
                knn = faiss.read_index(str(path / f"knn.{i_clst}.{r}"))
                self.data[int(i_clst)] = (knn, cc)
        return super().sync(**kwargs)

    def update(
        self,
        dl: DataLoader,
        y_true: ArrayLike,
        y_clst: ArrayLike,
        matching: Matching,
    ) -> None:
        """
        Reminder:
            `dl` has to iterate over the whole dataset, even if this method is
            called in a distributed environment. The labels vectors must also
            cover the whole dataset.
        """
        self.matching = to_int_matching(matching)
        y_clst = to_int_array(y_clst)
        n_features = next(iter(dl))[0].flatten(1).shape[-1]
        p1, p2, _, _ = otm_matching_predicates(
            y_true, y_clst, self.matching, c_a=self.n_classes
        )
        p_cc = (p1 & p2).sum(axis=0).astype(bool)  # (n_samples,)

        # Cluster labels that this rank has to manage
        clsts = self._distribute_labels(y_clst)
        # ↓ i_clst -> (knn idx, list of batches CC samples in this clst)
        data: dict[int, tuple[faiss.IndexHNSWFlat, list[Tensor]]] = {
            i_clst: (faiss.IndexHNSWFlat(n_features, self.k), [])
            for i_clst in clsts
        }

        tqdm, n_seen = make_tqdm(self.tqdm_style), 0
        for z, *_ in tqdm(dl, f"Building {len(data)} KNN indices"):
            z = z.flatten(1)  # (bs, n_feat.)
            _y_clst = y_clst[n_seen : n_seen + len(z)]  # (bs,)
            _p_cc = p_cc[n_seen : n_seen + len(z)]  # (bs,)
            for i_clst in np.unique(_y_clst):
                if i_clst not in data:
                    continue  # Cluster not managed by this rank
                # ↓ Mask for smpls in this batch that are CC and in i_clsts
                _p_cc_i_clst = _p_cc & (_y_clst == i_clst)
                if not _p_cc_i_clst.any():
                    continue  # No CC sample in cluster i_clst in this batch
                _z = z[_p_cc_i_clst]
                data[i_clst][0].add(to_array(_z).astype(np.float32))
                data[i_clst][1].append(_z)
            n_seen += len(z)

        # ↓ i_clst -> (knn idx, tensor of all CC samples in this clst)
        #   IF i_clst has at least one CC sample
        self.data = {
            i_clst: (idx, torch.cat(lst))
            for i_clst, (idx, lst) in data.items()
            if lst
        }
