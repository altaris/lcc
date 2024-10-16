"""Louvain clustering specific stuff"""

from typing import Any, Iterator, Literal

import faiss
import faiss.contrib.torch_utils
import networkx as nx
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader

from ..utils import check_cuda, make_tqdm, to_array


def louvain_communities(
    dl: DataLoader,
    k: int,
    key: str | None = None,
    device: Any = None,
    tqdm_style: Literal["notebook", "console", "none"] | None = None,
) -> tuple[list[set[int]], np.ndarray]:
    """
    Returns louvain communities of a set of points, iterated through a torch
    dataloader.

    Args:

        dl (DataLoader): The dataset dataloader.
        k (int, optional): The number of neighbors to consider for the Louvain
            clustering algorithm. Note that a point is not considered as one if
            its nearest neighbors.
        key (str | None, optional): The key to use to extract the data from the
            dataloader batches. If left to `None`, batches are assumed to be
            tensors. Otherwise, they are assumed to be dictionaries and the
            actual tensor is located at that key.
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

    use_cuda, device = check_cuda(device)

    def _batches(desc: str | None = None) -> Iterator[Tensor]:  # shorthand
        if desc:
            everything = make_tqdm(tqdm_style)(dl, desc, leave=False)
        else:
            everything = dl
        for x in everything:
            if key is not None:
                x = x[key]
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
                # [(n + j, int(i), np.exp(-d / np.sqrt(pca_dim))) for i, d in a]
                # had numerical stability issues with above weights
                [(n + j, int(i), 1) for i, _ in a]
            )
        n += len(batch)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    # Reciprocal edges share the same weight, so it's ok to discard one of them
    # See
    # https://networkx.org/documentation/stable/reference/classes/generated/networkx.DiGraph.to_undirected.html
    graph = nx.to_undirected(graph)

    communities: list[set[int]] = nx.community.louvain_communities(
        graph,
        **({"backend": "cugraph"} if use_cuda else {}),
    )
    y_louvain = [0] * n
    for i, c in enumerate(communities):
        for n in c:
            y_louvain[n] = i
    return communities, np.array(y_louvain)
