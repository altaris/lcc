"""Choice of classes for correction"""

from functools import partial

import networkx as nx
import numpy as np
from scipy.sparse import coo_array
from torch import Tensor
from torchmetrics.functional.classification import multiclass_confusion_matrix
from tqdm import tqdm


def confusion_graph(
    y_pred: Tensor, y_true: Tensor, n_classes: int
) -> nx.Graph:
    """
    Create a confusion graph from predicted and true labels. Nodes are labels
    and edges' weight are the number of times two labels are confused for each
    other.

    Args:
        y_pred (Tensor): A `(N,)` int tensor or an `(N, C)`
            probabilities/logits float tensor
        y_true (Tensor): A `(N,)` int tensor
        n_classes (int):

    Warning:
        There are no loops, i.e. correct predictions are not reported in the
        graph unlike in usual confusion matrices.
    """
    cm = multiclass_confusion_matrix(y_pred, y_true, num_classes=n_classes)
    cm = cm + cm.T  # Don't care about diagonal elements
    scm, cg = coo_array(cm), nx.Graph()
    for i, j, w in zip(scm.row, scm.col, scm.data):
        if i != j:
            cg.add_edge(i, j, weight=w)
    return cg


def heaviest_connected_subgraph(
    graph: nx.Graph,
    max_size: int | None = None,
    strict: bool = False,
    key: str = "weight",
) -> tuple[nx.Graph, float]:
    """
    Find the heaviest connected full subgraph of an undirected graph.

    Under the hood, this function maintains a list of connected full subgraphs
    and iteratively adds the heaviest edge to those subgraphs that have one if
    its endpoints. Note that:
    * if no graph touch the edge, then it is added to the list as its own
      subgraph;
    * if a graph have both endpoints, then the edge was already part of that
      graph and it is not modified;
    * graphs in the list that have already reached `max_size` are not modified.
    Finally, the heaviest graph is returned.

    Args:
        graph (nx.Graph):
        max_size (int | None, optional): If left to `None`, returns the
            heaviest connected component.
        strict (bool, optional): If `True`, the returned graph is guaranteed to
            have exactly `max_size` nodes. If `False`, the returned graph may
            have fewer (but never more) nodes.
        key (str, optional): The edge attribute to use as weight.

    Returns:
        A connected subgraph and its total weight (see also `total_weight`).

    Warning:
        Setting `strict` to `True` can make the problem impossible, e.g. if
        `graph` doesn't have a large enough connected component. In such cases,
        a `RuntimeError` is raised.
    """
    _total_weight = partial(total_weight, key=key)
    if max_size is None:
        subgraphs = sorted(
            map(graph.subgraph, nx.connected_components(graph)),
            key=_total_weight,
            reverse=True,
        )
        return subgraphs[0]
    edges = sorted(
        graph.edges(data=True), key=lambda e: e[2][key], reverse=True
    )
    subgraphs = []
    for u, v, _ in tqdm(edges, desc="Finding heaviest subgraph", leave=False):
        if not subgraphs:
            subgraphs.append(graph.subgraph([u, v]))
            continue
        has_u = np.array([g.has_node(u) for g in subgraphs], dtype=bool)
        has_v = np.array([g.has_node(v) for g in subgraphs], dtype=bool)
        if np.any(has_u & has_v):  # a graph already contains edge (u, v)
            continue
        p = (has_u | has_v) & (np.array(list(map(len, subgraphs))) < max_size)
        to_extend = [g for i, g in enumerate(subgraphs) if p[i]]
        subgraphs = [g for i, g in enumerate(subgraphs) if ~p[i]]
        if to_extend:
            for g in to_extend:
                subgraphs.append(graph.subgraph(list(g.nodes) + [u, v]))
        else:
            subgraphs.append(graph.subgraph([u, v]))
    if strict:
        subgraphs = list(filter(lambda g: len(g) >= max_size, subgraphs))
    subgraphs = sorted(
        subgraphs,
        key=_total_weight,
        reverse=True,
    )
    if not subgraphs:
        raise RuntimeError(
            "Could not find heaviest subgraph with given size constraint. "
            "Try again with strict=False."
        )
    return subgraphs[0], _total_weight(subgraphs[0])


def total_weight(graph: nx.Graph, key: str = "weight") -> int:
    """Simple helper to compute the total edge weight of a graph."""
    return sum(d[key] for _, _, d in graph.edges(data=True))


def max_connected_confusion_choice(
    y_pred: Tensor, y_true: Tensor, n_classes: int, n: int
) -> tuple[list[int], int]:
    """
    Chooses the classes that are most confused for each other according to
    some confusion graph scheme.

    Args:
        y_pred (Tensor): A `(N,)` int tensor or an `(N, C)`
            probabilities/logits float tensor
        y_true (Tensor): A `(N,)` int tensor
        n_classes (int): Number of classes in the whole dataset
        n (int): Number of classes to choose

    Returns:
        An `int` list of `n` classes **or less**, and the total number of
        confused sample number along these classes.
    """
    cg = confusion_graph(y_pred, y_true, n_classes)
    hcg, w = heaviest_connected_subgraph(cg, max_size=n, strict=False)
    return Tensor(list(hcg.nodes)).int().tolist(), int(w)


# def uniform_choice(
#     a: Tensor,
#     n: int | None = None,
#     generator: torch.Generator | None = None,
# ) -> Tensor:
#     """
#     Analogous to
#     [`numpy.random.choice`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html)
#     except the selection is without replacement the selection distribution is
#     uniform.

#     Args:
#         a (Tensor): Tensor to sample from.
#         n (int | None, optional): Number of samples to draw. If `None`, returns
#             a permutation of `a`
#         generator (torch.Generator | None, optional):
#     """
#     idx = torch.randperm(len(a), generator=generator)
#     return a[idx] if n is None else a[idx[:n]]
