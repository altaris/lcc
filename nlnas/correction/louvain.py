"""Louvain clustering specific stuff"""

import warnings
from collections import defaultdict
from typing import Any, Iterator, Literal

import faiss
import faiss.contrib.torch_utils
import networkx as nx
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader

from ..utils import check_cuda, make_tqdm, to_array


def _louvain_or_leiden(graph: nx.Graph, device: Any = None) -> list[set[int]]:
    """
    Explicit dispatch to
    [`nx.community.louvain_communities`](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.louvain.louvain_communities.html)
    or
    [`cugraph.leiden`](https://docs.rapids.ai/api/cugraph/stable/api_docs/api/cugraph/cugraph.leiden)
    depending on the availability of CUDA and the requested device.

    Args:
        graph (nx.Graph):
        device (Any, optional):
    """
    use_cuda, device = check_cuda(device)
    if use_cuda:
        import cugraph

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Note: since cugraph.leiden receives a networkx object, it returns
            # tuple[dict[int, int], float]. This behavior isn't documented but
            # the implementation is clear
            # https://github.com/rapidsai/cugraph/blob/add69b8e2a63f4158172f78461f2f9703cbe8eaf/python/cugraph/cugraph/community/leiden.py#L126
            ni_to_ci: dict[int, int] = cugraph.leiden(graph)[0]
            # si = node index, ci = community index
        ci_to_nis = defaultdict(list)
        for ni, ci in ni_to_ci.items():
            ci_to_nis[ci].append(ni)
        return [set(v) for v in ci_to_nis.values()]
    return nx.community.louvain_communities(graph)  # type: ignore


def louvain_communities(
    dl: DataLoader,
    k: int,
    device: Any = "cpu",
    tqdm_style: Literal["notebook", "console", "none"] | None = None,
) -> tuple[list[set[int]], np.ndarray]:
    """
    Returns louvain communities of a set of points, iterated through a torch
    dataloader.

    Args:

        dl (DataLoader): A dataloader that yields tensors.
        k (int, optional): The number of neighbors to consider for the Louvain
            clustering algorithm. Note that a point is not considered as one if
            its nearest neighbors.
        device (Any, optional): If left to `None`, uses CUDA if it is available,
            otherwise falls back to CPU. Setting `cuda` while CUDA isn't
            available will **silently** fall back to CPU.
        tqdm_style (Literal['notebook', 'console', 'none'] | None, optional):

    Returns:
        1. (`list[set[int]]`) The actual louvain communities, which is a
           partition of the set $\\\\{ 0, 1, ..., N-1 \\\\}$.
        2. (`np.ndarray`) The `(N,)` label vector for the communities. Let's
           call it `y_louvain`. If there are `c` communities, then `y_louvain`
           has integer values in $\\\\{ 0, 1, ..., c-1 \\\\}$, and if
           `y_louvain[i] == j`, then `z[i]` belongs to the $j$-th community
    """

    def _batches(desc: str | None = None) -> Iterator[Tensor]:  # shorthand
        if desc:
            everything = make_tqdm(tqdm_style)(dl, desc, leave=False)
        else:
            everything = dl
        for x in everything:
            yield x.flatten(1).to(device)

    z = next(_batches())
    z = z.flatten(1)
    n_features = z.shape[-1]

    index = faiss.IndexHNSWFlat(n_features, k)
    for batch in _batches(f"Building KNN index (k={k})"):
        u = to_array(batch).astype(np.float32)
        index.add(u)

    graph, n = nx.DiGraph(), 0
    for batch in _batches(f"Building KNN graph (k={k})"):
        u = to_array(batch).astype(np.float32)
        dst, idx = index.search(u, k + 1)
        for j, all_i, all_d in zip(range(len(idx)), idx, dst):
            a = list(zip(all_i, all_d))[1:]  # exclude self as nearest neighbor
            graph.add_weighted_edges_from(
                [(n + j, int(i), 1) for i, _ in a]
                # Had numerical stability issues with
                # [(n + j, int(i), np.exp(-d / np.sqrt(pca_dim))) for i, d in a]
            )
        n += len(batch)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    # Reciprocal edges share the same weight, so it's ok to discard one of them
    # See
    # https://networkx.org/documentation/stable/reference/classes/generated/networkx.DiGraph.to_undirected.html
    graph = nx.to_undirected(graph)

    communities = _louvain_or_leiden(graph, device)
    y_louvain = [0] * n
    for i, c in enumerate(communities):
        for n in c:
            y_louvain[n] = i
    return communities, np.array(y_louvain)
