"""Choice of classes for correction"""

from functools import partial
from itertools import combinations
from typing import Literal, TypeAlias

import networkx as nx
import numpy as np
from loguru import logger as logging
from torch import Tensor
from torchmetrics.functional.classification import multiclass_confusion_matrix
from tqdm import tqdm

from ..utils import to_tensor

LCC_CLASS_SELECTIONS: list[str] = [
    # TODO: Re-enable support for non-trivial class selection policies
    # "top_pair_1",
    # "top_pair_5",
    # "top_pair_10",
    # "top_connected_2",
    # "top_connected_5",
    # "top_connected_10",
    # "max_connected",
]
"""
Supported non-trivial class selection policies for LCC. See
`nlnas.correction.LCCClassSelection`.
"""

LCCClassSelection: TypeAlias = Literal[
    # "all",  # Use `None` instead
    "top_pair_1",
    "top_pair_5",
    "top_pair_10",
    "top_connected_2",
    "top_connected_5",
    "top_connected_10",
    "max_connected",
]
"""
Non-trivial (true) class selection policies for LCC. This is used to determine
which samples will undergo LCC. On top of the `None` policy (which selects all
samples), the following are supported:
- `top_pair_<N>`: Consider the top `N` confusion pairs.  This is **not**
  necessarily a "connected" choice, i.e. it is possible that not every two
  classes are confused with each other. For example, policy `top_pair_2` may
  choose classes, say, `0`, `1`, `2`, `3`, with `(0, 1)` being the top confused
  pair, `(2, 3)` being the second most confused pair, but where the pairs `(0,
  2)`, `(0, 3)`, `(1, 2)` and `(1, 3)` are not confused at all. Also, note that
  the this choice may return less than `2 * N` classes. This happens if the same
  class is in more than one top `N` confusion pair.
- `top_connected_<N>`: Considers the heaviest `N` (or less) node connected
  subgraph in the confusion graph. Note that `top_connected_2` is equivalent to
  `top_pair_1`.
- `max_connected`: Consider the largest connected component of the confusion
  graph (see `nlnas.correction.choice.max_connected_confusion_choice` and
  `nlnas.correction.choice.confusion_graph`).
"""


class GraphTotallyDisconnected(ValueError):
    """
    Raised in `nlnas.correction.heaviest_connected_subgraph` when a graph is
    totally disconnected (has no edges).
    """


def confusion_graph(
    y_pred: Tensor | np.ndarray | list[int],
    y_true: Tensor | np.ndarray | list[int],
    n_classes: int,
    threshold: int = 10,
) -> nx.Graph:
    """
    Create a confusion graph from predicted and true labels. Two labels $a$ and
    $b$ are confused if there is at least one sample belonging to class $a$ that
    is predicted as class $b$, or vice versa. The nodes of the confusion graph
    are labels, two labels are connected by an edge if they are confused by at
    least `threshold` samples, and the edges' weight are the number of times two
    labels are confused for each other.

    Args:
        y_pred (Tensor | np.ndarray): A `(N,)` int tensor or an `(N, n_classes)`
            probabilities/logits float tensor
        y_true (Tensor | np.ndarray): A `(N,)` int tensor
        n_classes (int):
        threshold (int, optional): Minimum number of times two classes must be
            confused (in either direction) to be included in the graph.

    Warning:
        There are no loops, i.e. correct predictions are not reported in the
        graph unlike in usual confusion matrices.
    """
    y_pred, y_true = to_tensor(y_pred), to_tensor(y_true)
    cm = multiclass_confusion_matrix(y_pred, y_true, num_classes=n_classes)
    cm = cm + cm.T  # Confusion in either direction
    cg = nx.Graph()
    for i, j in combinations(list(range(n_classes)), 2):
        if (w := cm[i, j].item()) >= threshold:
            cg.add_edge(i, j, weight=w)
    return cg


def choose_classes(
    y_true: Tensor | np.ndarray | list[int],
    y_pred: Tensor | np.ndarray | list[int],
    policy: LCCClassSelection | Literal["all"] | None = None,
) -> list[int] | None:
    """
    Given true and predicted labels, select classes whose samples should undergo
    LCC based on some policy. See `nlnas.correction.LCCClassSelection`.

    For convenience, this method returns `None` if all classes should be
    considered.

    Warning:
        When selecting a `"top_<N>"` policy, the returned list may have fewer
        than `N` elements. For example, this happens when there are fewer than
        `N` classes in the dataset.
    """
    y_true, y_pred = to_tensor(y_true), to_tensor(y_pred)
    if policy is None:
        return None
    if policy == "all":
        logging.warning(
            "LCC class selection policy 'all' is deprecated. Use `None` instead (which has the same effect)"
        )
        return None
    n_classes = y_true.unique().numel()
    if policy.startswith("top_pair_"):
        n = int(policy[9:])
        if n >= n_classes:
            return None
        pairs = top_confusion_pairs(y_pred, y_true, n_classes, n_pairs=n)
        return list(set(sum(pairs, ())))
    try:
        if policy.startswith("top_connected_"):
            n = int(policy[14:])
            if n >= n_classes:
                return None
            return max_connected_confusion_choice(
                y_pred, y_true, n_classes, n
            )[0]
        return max_connected_confusion_choice(y_pred, y_true, n_classes)[0]
    except GraphTotallyDisconnected:
        return None


def heaviest_connected_subgraph(
    graph: nx.Graph,
    max_size: int | None = None,
    strict: bool = False,
    key: str = "weight",
) -> tuple[nx.Graph, float]:
    """
    Find the heaviest connected full subgraph of an undirected graph with
    weighted edges. In other words, returns the connected component whose total
    edge weight is the largest.

    Under the hood, this function maintains a list of connected full subgraphs
    and iteratively adds the heaviest edge to those subgraphs that have one if
    its endpoints. Note that:
    - if no graph touch the current edge, then it is added to the list as its
      own subgraph;
    - if a graph have both endpoints of the current edge, then the edge was
      already part of that graph and it is not modified;
    - graphs in the list that have already reached `max_size` are not modified.

    Finally, the heaviest graph is returned.

    Warning:
        Setting `strict` to `True` can make the problem impossible, e.g. if
        `graph` doesn't have a large enough connected component. In such cases,
        a `RuntimeError` is raised.

    Warning:
        If the graph is totally disconnected (i.e. has no edges), then a
        `GraphTotallyDisconnected` exception is raised, rather than returning a
        subgraph with a single node.

    Args:
        graph (nx.Graph): Most likely the confusion graph returned by
            `nlnas.choice.confusion_graph` eh?
        max_size (int | None, optional): If left to `None`, returns the
            heaviest connected component.
        strict (bool, optional): If `True`, the returned graph is guaranteed to
            have exactly `max_size` nodes. If `False`, the returned graph may
            have fewer (but never more) nodes.
        key (str, optional): The edge attribute to use as weight.

    Returns:
        A connected subgraph and its total weight (see also `total_weight`).
    """
    if not graph.edges:
        raise GraphTotallyDisconnected()
    _total_weight = partial(total_weight, key=key)
    if max_size is None:
        subgraphs = sorted(
            map(graph.subgraph, nx.connected_components(graph)),
            key=_total_weight,
            reverse=True,
        )
        return subgraphs[0], _total_weight(subgraphs[0])
    edges = sorted(
        graph.edges(data=True), key=lambda e: e[2][key], reverse=True
    )
    subgraphs = []
    for u, v, _ in tqdm(edges, desc="Finding heaviest subgraph"):
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
    y_pred: Tensor | np.ndarray | list[int],
    y_true: Tensor | np.ndarray | list[int],
    n_classes: int,
    n: int | None = None,
    threshold: int = 0,
) -> tuple[list[int], int]:
    """
    Chooses the classes that are most confused for each other according to
    some confusion graph scheme.

    Args:
        y_pred (Tensor): A `(N,)` int tensor or an `(N, n_classes)`
            probabilities/logits float tensor
        y_true (Tensor): A `(N,)` int tensor
        n_classes (int): Number of classes in the dataset.
        n (int, optional): Number of classes to choose. If `None`, returns the
            classes in the largest connected component of the confusion graph.
        threshold (int, optional): Ignore pairs of classes that are confused by
            less than that number of samples. See also
            `nlnas.choice.confusion_graph`.

    Returns:
        An `int` list of `n` classes **or less**, and the total number of
        confused sample number along these classes.
    """
    cg = confusion_graph(y_pred, y_true, n_classes, threshold=threshold)
    hcg, w = heaviest_connected_subgraph(cg, max_size=n, strict=False)
    return Tensor(list(hcg.nodes)).int().tolist(), int(w)


def top_confusion_pairs(
    y_pred: Tensor | np.ndarray | list[int],
    y_true: Tensor | np.ndarray | list[int],
    n_classes: int,
    n_pairs: int | None = None,
    threshold: int = 0,
) -> list[tuple[int, int]]:
    """
    Returns the top `n_pairs` top pairs of labels that exhibit the most
    confusion. The confusion between two labels $a$ and $b$ is the number of
    samples in true class $a$ that are predicted as class $b$, plus the number
    of samples in true class $b$ that are predicted as class $a$.

    Example:
        >>> y_pred, y_true = [0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 0]
        >>> top_confusion_pairs(y_pred, y_true, n_classes=3, n_pairs=2)
        [(1, 2), (0, 2)]

    Args:
        y_pred (Tensor): A `(N,)` int tensor or an `(N, n_classes)`
            probabilities/logits float tensor
        y_true (Tensor): A `(N,)` int tensor
        n_classes (int): Number of classes in the dataset
        n_pairs (int | None, optional): Number of desired pairs. The actual
            result might have less. If `None`, returns all pairs of labels that
            have at lease `threshold` confused samples.
        threshold (int, optional): Minimum number of confused samples between a
            pair of labels to be included in the list

    Returns:
        The top `n_pairs` pairs **or less** of labels that exhibit the most
        confusion.
    """
    y_pred, y_true = to_tensor(y_pred), to_tensor(y_true)
    cm = multiclass_confusion_matrix(y_pred, y_true, n_classes).numpy()
    cm = cm + cm.T  # Confusion in either direction
    cm = cm * (1 - np.eye(len(cm)))  # Remove the diagonal
    idx = cm.argsort(axis=None)  # Flat indices
    idx = np.flip(idx)
    cp = np.stack(np.unravel_index(idx, cm.shape)).T
    lst = [(i, j) for i, j in cp if cm[i, j] > threshold and i < j]
    return lst if n_pairs is None else lst[:n_pairs]
