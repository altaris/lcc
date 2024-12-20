"""
Peer pressure graph clustering

TODO:
    CSC format seems to be faster than CSR. (about 30s gain on ImageNet logits
    with `k = 100`)
"""

import networkx as nx
import numpy as np
from lightning_fabric import Fabric
from pytorch_lightning.strategies import Strategy
from scipy.sparse import csr_array, eye_array

from ..datasets.batched_tensor import BatchedTensorDataset
from ..utils import TqdmStyle, make_tqdm
from .knn import knn_graph
from .utils import EarlyStoppingLoop


def _ppc(
    a: csr_array,
    max_iter: int = 100,
    patience: int = 10,
    score_delta: float = 1e-3,
    min_n_moves: int = 0,
    min_r_moves: float = 0.0,
    tqdm_style: TqdmStyle = None,
) -> tuple[np.ndarray, float]:
    """
    Peer pressure graph clustering with weighted mean cluster conductance as an
    objective function.

    Args:
        a (csr_array): Sparse square adjacency matrix (as a `csr_array`).
        max_iter (int, optional): Maximum number of peer pressure iterations.
        patience (int, optional): Early stopping if mean conductance does not
            improve after this many iterations
        score_delta (float, optional): Minimum mean conductance improvement for
            a clustering to be considered better than the previous best.
        min_n_moves (int, optional): Early stopping if the number of nodes
            moving to another cluster is less than this.
        min_r_moves (float, optional): Early stopping if the ratio of nodes
            moving to another cluster is less than this.

    Returns:
        A `(n_nodes,)` int label array and the corresponding mean conductance
        score.
    """
    n_nodes, n_edges = a.shape[0], a.size / 2
    b = eye_array(n_nodes, format=a.format)
    # assert isinstance(b, csr_array)  # For typechecking
    loop = EarlyStoppingLoop(max_iter, patience, score_delta)
    loop.propose(b, float("inf"))
    tqdm = make_tqdm(tqdm_style)
    for _ in tqdm(loop, desc="Peer pressure clustering"):
        new_b = _ppi(a, b)
        n_moves = (b - new_b).power(2).sum() / 2
        r_moves = n_moves / n_nodes * 100
        if n_moves < min_n_moves or r_moves < min_r_moves:
            break
        b = new_b
        ab = a @ b
        volume = 2 * ab.sum(0)
        cut = volume / 2 - (b.T @ ab).diagonal()
        p = volume > 0
        if (n_clst := p.sum()) == 0:  # WTF this is not possible
            raise RuntimeError("All clusters are empty which shouldn't happen")
        if n_clst == 1:  # Only one cluster, no need to continue peer pressure
            loop.propose(b, 0)
            break
        # If there is at least 2 clusters, then they must all have non-zero
        # volume and co-volume, so conductance is well-defined
        volume, cut = volume[p], cut[p]
        conductance = cut / np.minimum(volume, 2 * n_edges - volume)
        score = np.average(conductance, weights=volume)
        loop.propose(b, score)
    b, score = loop.best()
    y_clst = b[:, b.sum(0) > 0].argmax(1).reshape(-1)
    return y_clst, score


def _ppi(a: csr_array, b: csr_array) -> csr_array:
    """
    Single peer pressure iteration, where nodes move to the cluster which is
    most popular with its neighbors.

    Args:
        a (csr_array): A sparse adjacency matrix
        b (csr_array): A clustering matrix of shape `(n_nodes, n_clusters)`,
            where `b[i, j] = 1` if node `i` is in cluster `j`, and `0`
            otherwise

    Returns:
        A new clustering matrix
    """
    col = (a @ b).argmax(axis=1).reshape(-1)
    row = np.arange(len(col))
    dat = np.ones_like(row)
    return csr_array((dat, (row, col)), shape=b.shape, dtype=np.int8)


def peer_pressure_clustering(
    ds: BatchedTensorDataset,
    k: int,
    strategy: Strategy | Fabric | None = None,
    n_features: int | None = None,
    tqdm_style: TqdmStyle = None,
) -> np.ndarray:
    """
    Nearest-neighbor peer pressure clustering with weighted mean cluster
    conductance as an objective function.

    Args:
        ds (BatchedTensorDataset):
        k (int):
        strategy (Strategy | Fabric | None, optional): Defaults to `None`,
            meaning that the algorithm will not be parallelized.
        n_features (int | None, optional):
        tqdm_style (TqdmStyle, optional):
    """
    graph = knn_graph(
        ds, k, strategy=strategy, n_features=n_features, tqdm_style=tqdm_style
    )
    a = nx.to_scipy_sparse_array(graph, format="csr")
    y_clst, _ = _ppc(a, tqdm_style=tqdm_style)
    return y_clst
