# pylint: disable=ungrouped-imports

"""Louvain clustering specific stuff"""

from typing import Literal

import networkx as nx
import numpy as np
import torch
from sklearn.base import TransformerMixin
from torch import Tensor

from .clustering import DEFAULT_K


def louvain_communities(
    z: np.ndarray | Tensor,
    k: int = DEFAULT_K,
    scaling: (
        Literal["standard", "minmax"] | TransformerMixin | None
    ) = "standard",
    device: Literal["cpu", "cuda"] | None = None,
) -> tuple[list[set[int]], np.ndarray]:
    """
    Returns louvain communities of a set of points.

    Args:
        z (np.ndarray | Tensor): A `(N, d)` array. If it has another
            shape, it is flattened to `(len(z), ...)`.
        k (int, optional): The number of neighbors to consider for the Louvain
            clustering algorithm, excluding self (i.e. a point is not
            considered as one if its nearest neighbors)
        scaling (Literal["standard", "minmax"] | TransformerMixin | None,
            optional): Scaling method for `z`. `"standard"` uses
            [`sklearn.preprocessing.StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
            or CUML equivalent, "minmax" uses
            [sklearn.preprocessing.MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
            or CUML equivalent. It is also possible to pass an actual class
            that has a `fit_transform` method with a single mandatory argument.
        device (Literal["cpu", "cuda"] | None, optional): If left to `None`,
            uses CUDA if it is available, otherwise falls back to CPU. Setting
            `cuda` while CUDA isn't available will silently fall back to CPU.

    Returns:
        1. (`list[set[int]]`) The actual louvain communities, which is a
           partition of the set $\\\\{ 0, 1, ..., N-1 \\\\}$.
        2. (`np.ndarray`) The `(N,)` label vector for the communities. Let's
           call it `y_louvain`. If there are `c` communities, then `y_louvain`
           has integer values in $\\\\{ 0, 1, ..., c-1 \\\\}$, and if
           `y_louvain[i] == j`, then `z[i]` belongs to the `j`-th community
    """
    use_cuda = (
        device == "cuda" or device is None
    ) and torch.cuda.is_available()
    if use_cuda:
        from cuml.neighbors import NearestNeighbors
        from cuml.preprocessing import MinMaxScaler, StandardScaler
    else:
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

    if scaling == "standard":
        scaling = StandardScaler()
    elif scaling == "minmax":
        scaling = MinMaxScaler()
    if isinstance(z, Tensor):
        z = z.cpu().detach().numpy()
    z = z.reshape(len(z), -1)
    z = z if scaling is None else scaling.fit_transform(z)  # type: ignore
    assert isinstance(z, np.ndarray)  # for typechecking

    index = NearestNeighbors(n_neighbors=min(k + 1, z.shape[0]))
    index.fit(z)
    adj = index.kneighbors_graph(z)
    graph = nx.from_scipy_sparse_array(adj, edge_attribute="weight")
    graph.remove_edges_from(nx.selfloop_edges(graph))  # exclude self as NN
    communities: list[set[int]] = nx.community.louvain_communities(  # type: ignore
        graph,
        **({"backend": "cugraph"} if use_cuda else {}),  # type: ignore
    )
    y_louvain = [0] * len(graph)
    for i, c in enumerate(communities):
        for n in c:
            y_louvain[n] = i
    return communities, np.array(y_louvain)


# def louvain_loss(
#     z: Tensor,
#     y_true: np.ndarray | Tensor,
#     k: int = DEFAULT_K,
#     n_true_classes: int | None = None,
#     device: Literal["cpu", "cuda"] | None = None,
# ) -> Tensor:
#     """
#     Calls `nlnas.correction.clustering.clustering_loss` with the Louvain
#     clustering data.
#     """
#     if isinstance(y_true, Tensor):
#         y_true = y_true.cpu().detach().numpy()
#     assert isinstance(y_true, np.ndarray)  # For typechecking
#     _, y_louvain = louvain_communities(z)
#     y_louvain = y_louvain[: len(y_true)]
#     # TODO: Why is y_louvain sometimes longer than y_true?
#     # This seems to only happen if z has nan's, in which case
#     # len(y_louvain) = len(y_true) + 1
#     matching = class_otm_matching(y_true, y_louvain)
#     return clustering_loss(
#         z, y_true, y_louvain, matching, k, n_true_classes, device
#     )
