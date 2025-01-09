"""Distributed KNN graph construction."""

import faiss
import networkx as nx
import numpy as np
import torch
from lightning_fabric import Fabric
from pytorch_lightning.strategies import ParallelStrategy, Strategy
from torch import Tensor
from torch.utils.data import DataLoader

from ..datasets.batched_tensor import BatchedTensorDataset
from ..logging import r0_warning
from ..utils import TqdmStyle, make_tqdm, to_array, to_tensor


def _edges(sources: Tensor, targets: Tensor) -> Tensor:
    """
    Given two tensors
    * `sources` of shape `(n,)`, and
    * `targets` of shape `(n, k)`,
    returns a tensor `e` of shape `(n * k, 2)` that contains all tuples
    `[sources[i], targets[i, j]]`, where $$0 \\leq i < n$$ and $$0 \\leq j <
    k$$.
    """
    k = targets.shape[-1]
    sources = sources.repeat(k, 1).T  # (n, k)
    return torch.stack([sources.flatten(), targets.flatten()], dim=-1)


def _to_array(u: Tensor) -> np.ndarray:
    """Convenience function for the specific needs of this module"""
    return to_array(u.flatten(1)).astype(np.float32)


def _weird_index(a: Tensor, b: Tensor) -> Tensor:
    """
    Given two tensors `a` of shape `(n, da)` and `b` of shape `(n, db)`, such
    that $$d_a \\leq d_b$$, returns a tensor `c` of shape `(n, db)` such that
    `c[i, j] = a[i, b[i, j]]`.
    """
    n, db = b.shape
    r = torch.arange(n).unsqueeze(1).expand(n, db)
    # r: (n, db) = [[0, ..., 0], ..., [n-1, ..., n-1]]
    return a[r, b]


def knn_graph(
    ds: BatchedTensorDataset,
    k: int,
    strategy: Strategy | Fabric | None = None,
    n_features: int | None = None,
    tqdm_style: TqdmStyle = None,
) -> nx.Graph:
    """
    Potentially distributed symmetric KNN graph construction. The dataloader
    must tield tuples of at least two tensors:
    * a 2D tensor of samples, if `n_features` is specified, then the last
      dimension must be `n_features`,
    * a 1D tensor of absolute indices of said samples in the dataset,
    * the rest is ignored.

    Warning:
        The dataloader `dl` must be unsharded (i.e. iterate over the whole
        dataset). If the function is called in a distributed environment, the
        dataloader's underlying dataset will be accessed and a new dataloader
        with distributed sampling will be constructed for part of the algorithm.

    TODO:
        There's a little optimization possible here.

        Every `faiss.IndexFlatL2` are constructed fo find $$k + 1$$ neighbors
        because we do not consider a sample to be its own neighbor, whereas
        `faiss` does. So in the first loop, we construct a $$(k + 1)$$-NN index,
        and in the second loop, we query it and gather the results from every
        rank.

        HOWEVER, if there is at least 2 ranks, we can get away with fitting and
        querying $$k$$-NN indices instead of $$(k + 1)$$-NN indices. The reason
        is that after gathering the queries, each sample will see `k *
        world_size >= k + 1` candidate neighbors. So after exluding the sample
        as its own candidate neighbor, we will have at least `k` candidates left
        which is enough.

    Args:
        dl (DataLoader): Unsharded dataloader
        k (int):
        strategy (Strategy | Fabric | None, optional): Defaults to `None`,
            meaning that the algorithm will not be parallelized.
        n_features (int | None, optional): The latent dimension of the samples.
            If left to `None`, it is inferred from the first batch of `dl`
        tqdm_style (TqdmStyle, optional):
    """
    if strategy is not None and not isinstance(
        strategy, (ParallelStrategy, Fabric)
    ):
        r0_warning(
            "Passed a strategy object to knn_graph, but strategy is not "
            "parallel. Falling back to non-distributed implementation."
        )
        strategy = None
    gr = strategy.global_rank if strategy is not None else 0
    ws = strategy.world_size if strategy is not None else 1

    dl = DataLoader(ds.distribute(strategy), batch_size=256)
    n_features = n_features or next(iter(ds))[0].shape[-1]
    tqdm = make_tqdm(tqdm_style)
    index = faiss.IndexFlatL2(n_features)
    if torch.cuda.is_available():
        gpu = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(gpu, gr, index)
    _absolute_indices: list[torch.Tensor] = []
    for z, idx, *_ in tqdm(dl, f"[Rank {gr}/{ws}] Building KNN index ({k=})"):
        _absolute_indices.append(idx)
        index.add(_to_array(z))
    absolute_indices = torch.cat(_absolute_indices, dim=0)

    graph = nx.Graph()
    dl = DataLoader(ds, batch_size=256)
    for z, idx, *_ in tqdm(dl, f"[Rank {gr}/{ws}] Building KNN graph ({k=})"):
        nei_dst, nei_idx = index.search(_to_array(z), k + 1)  # both (bs, k+1)
        nei_dst, nei_idx = to_tensor(nei_dst), to_tensor(nei_idx)
        nei_idx = absolute_indices[nei_idx]  # (bs, k+1)
        if strategy is not None and strategy.world_size > 1:
            if isinstance(strategy, ParallelStrategy):
                nei_dst = strategy.batch_to_device(nei_dst)
                nei_idx = strategy.batch_to_device(nei_idx)
            nei_dst = strategy.all_gather(nei_dst)  # (ws, bs, k+1)
            nei_idx = strategy.all_gather(nei_idx)  # (ws, bs, k+1)
            assert isinstance(nei_dst, Tensor)  # for typechecking
            assert isinstance(nei_idx, Tensor)  # for typechecking
            nei_dst = nei_dst.permute(1, 0, 2).flatten(1)  # (bs, ws * (k+1))
            nei_idx = nei_idx.permute(1, 0, 2).flatten(1)  # (bs, ws * (k+1))
        argsort = nei_dst.argsort(dim=1, descending=False)  # (bs, ws * (k+1))
        argsort = argsort[:, 1 : k + 1]  # (bs, k) Ignore self as KNN
        nei_idx = _weird_index(nei_idx, argsort)  # (bs, k)
        e = _edges(idx.to(nei_idx), nei_idx)  # (bs * k, 2)
        graph.add_edges_from(e.cpu().numpy())
    graph.remove_edges_from(nx.selfloop_edges(graph))  # Just to be sure
    return graph
