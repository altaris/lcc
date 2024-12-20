"""Louvain clustering specific stuff"""

import warnings
from collections import defaultdict
from typing import Any

import networkx as nx
import numpy as np
from lightning_fabric import Fabric
from pytorch_lightning.strategies import Strategy

from ..datasets.batched_tensor import BatchedTensorDataset
from ..utils import TqdmStyle, check_cuda
from .knn import knn_graph


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


def louvain_clustering(
    ds: BatchedTensorDataset,
    k: int,
    strategy: Strategy | Fabric | None = None,
    n_features: int | None = None,
    tqdm_style: TqdmStyle = None,
    device: Any = "cpu",
) -> np.ndarray:
    """
    Args:
        ds (BatchedTensorDataset):
        k (int):
        strategy (Strategy | Fabric | None, optional): Defaults to `None`,
            meaning that the algorithm will not be parallelized.
        n_features (int | None, optional):
        tqdm_style (TqdmStyle, optional):
        device (Any, optional):
    """
    graph = knn_graph(
        ds, k, strategy=strategy, n_features=n_features, tqdm_style=tqdm_style
    )
    communities = _louvain_or_leiden(graph, device)
    y_clst = [0] * graph.number_of_nodes()
    for i_clst, clst in enumerate(communities):
        for smpl in clst:
            y_clst[smpl] = i_clst
    return np.array(y_clst)
