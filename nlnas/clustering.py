# pylint: disable=ungrouped-imports
"""Clustering of latent representations"""

from itertools import product
from math import sqrt
from typing import Literal

import networkx as nx
import numpy as np
import torch
from sklearn.base import TransformerMixin
from torch import Tensor

if torch.cuda.is_available():
    from cuml.neighbors import NearestNeighbors
    from cuml.preprocessing import MinMaxScaler, StandardScaler
else:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import MinMaxScaler, StandardScaler


def _otm_matching(
    graph: nx.DiGraph,
    set_a: list[str],
    set_b: list[str],
    mode: Literal["min", "max"] = "min",
) -> dict[str, set[str]]:
    """
    Given a edge-weighted graph `graph` and two disjoint node subsets `set_a`
    and `set_b`, such that the full subgraph spanned by `set_a | set_b` is
    bipartite, finds an optimal one-to-many matching between nodes of `set_a`
    and nodes of `set_b`. A one-to-many matching is like a usual matching
    except that a node in `set_a` can match to any number of nodes in `set_b`.

    Under the hood, this method adds two nodes to (a copy of) `graph`, called
    supersource and supersink. The supersource is connected to every node in
    `set_a`, and every node in `set_b` is connected to the supersink. Using
    adequate node demands and edge capacities, the optimal flow from the
    supersource to the supersink gives the optimal one-to-many matching.

    Args:
        graph (nx.DiGraph): Edges in this graph must have a numerical `weight`
            tag.
        set_a (list[str]):
        set_b (list[str]):
        mode (Literal["min", "max"], optional):

    Returns:
        A dict that maps a node in `set_a` to the subset of its matched nodes
        in `set_b`. The keys of this dict is exactly `set_a` (up to order).

    Warning:
        `graph` must not have any node called `__src__` or `__tgt__`
    """
    flow_graph, src, tgt = graph.copy(), "__src__", "__tgt__"
    flow_graph.add_node(src, demand=-len(set_b))
    flow_graph.add_node(tgt, demand=len(set_b))
    flow_graph.add_edges_from([(src, a) for a in set_a])
    flow_graph.add_edges_from([(b, tgt, {"capacity": 1}) for b in set_b])
    for e in graph.edges:
        d = flow_graph.edges[e]
        d["capacity"] = 1
        if mode == "max" and "weight" in d:
            d["weight"] *= -1
    flow = nx.max_flow_min_cost(flow_graph, src, tgt)
    return {
        a: {b for b, v in flow_from_a.items() if (v != 0 and b != tgt)}
        for a, flow_from_a in flow.items()
        if a in set_a
    }


def class_otm_matching(
    y_a: np.ndarray, y_b: np.ndarray
) -> dict[int, set[int]]:
    """
    Let `y_a` and `y_b` be `(N,)` integer array. We think of them as classes on
    some dataset, say `x`, which we call respectively a-classes and b-classes.
    This method performs a one-to-many matching from the classes in `y_a` to
    the classes in `y_b` to overall maximize the cardinality of the
    intersection between a-classes and the union of their matched b-classes.

    Example:

        >>> y_a = np.array([ 1,  1,  1,  1,  2,  2,  3,  4,  4])
        >>> y_b = np.array([10, 50, 10, 20, 20, 20, 30, 30, 30])
        >>> otm_matching(y_a, y_b)
        {1: {10, 50}, 2: {20}, 3: set(), 4: {30}}

        Here, `y_a` assigns class `1` to samples 0 to 3, label `2` to samples 4
        and 5 etc. On the other hand, `y_b` assigns its own classes to the
        dataset. What is the best way to regroup classes of `y_b` to
        approximate the labelling of `y_a`? The `otm_matching` return value
        argues that classes `10` and `15` should be regrouped under `1` (they
        fit neatly), label `20` should be renamed to `2` (eventhough it "leaks"
        a little, in that sample 3 is labelled with `1` and `20`), and class
        `30` should be renamed to `4`. No class in `y_b` is assigned to class
        `3` in this matching.

    Note:
        The values in `y_a` and `y_b` don't actually need to be distinct: the
        following works fine

        >>> y_a = np.array([1, 1, 1, 1, 2, 2, 3, 4, 4])
        >>> y_b = np.array([1, 5, 1, 2, 2, 2, 3, 3, 3])
        >>> otm_matching(y_a, y_b)
        {1: {1, 5}, 2: {2}, 3: set(), 4: {3}}

    Args:
        y_a (np.ndarray): A `(N,)` integer array. Unlike in the examples above,
            it is best if it only contains values in $\\\\{ 0, 1, ..., c_a - 1
            \\\\}$ for some $c_a > 0$.
        y_b (np.ndarray): A `(N,)` integer array. Unlike in the examples above,
            it is best if it only contains values in $\\\\{ 0, 1, ..., c_b - 1
            \\\\}$ for some $c_b > 0$.
    """
    match_graph = nx.DiGraph()
    for i, j in product(np.unique(y_a), np.unique(y_b)):
        n = np.sum((y_a == i) & (y_b == j))
        match_graph.add_edge(f"a_{i}", f"b_{j}", weight=n)
    matching = _otm_matching(
        match_graph,
        [f"a_{i}" for i in np.unique(y_a)],
        [f"b_{i}" for i in np.unique(y_b)],
        mode="max",
    )
    return {
        int(a_i.split("_")[-1]): {int(b_j.split("_")[-1]) for b_j in b_js}
        for a_i, b_js in matching.items()
    }


def clustering_loss(
    z: Tensor,
    y_true: np.ndarray | Tensor,
    y_cl: np.ndarray | Tensor,
    matching: dict[int, set[int]] | dict[str, set[int]],
    k: int = 5,
) -> Tensor:
    """
    Mean distance between every missed point and the average distance to their
    correctly clustered k-nearest neighbors in the same class.

    In more details, let $a_i$ be the true class of $z_i$, and $\\\\{ b_{i, 1},
    \\\\ldots \\\\}$ be the cluster classes matched to the true class $a_i$. If
    $z_i$ is a "missed points", i.e. if the cluster class of $z_i$ is not among
    $\\\\{ b_{i, 1}, \\\\ldots \\\\}$, then it will contribute a term to the
    clustering loss, equal to the mean distance between $z_i$ and its k-nearest
    neighbors in class $a_i$ and that also belong to one of the matched cluster
    classes $b_{i, j}$ (i.e. k-nearest neighbors in the same class that are
    furthermore correctly clustered)

    The clustering loss is the average of these terms, divided by
    $\\\\sqrt{d}$, where $d$ is the dimension of the latent space.

    Args:
        z (Tensor):
        y_true (np.ndarray | Tensor):
        y_cl (np.ndarray | None, optional):
        matching (dict[int, set[int]] | dict[str, set[int]]):
        k (int, optional):
    """

    def _np(a: Tensor) -> np.ndarray:
        return a.cpu().detach().numpy()

    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().detach().numpy()
    if isinstance(y_cl, Tensor):
        y_cl = y_cl.cpu().detach().numpy()
    assert isinstance(y_true, np.ndarray)  # For typechecking
    assert isinstance(y_cl, np.ndarray)  # For typechecking
    matching = {int(a): bs for a, bs in matching.items()}
    p1, p2, p3, _ = otm_matching_predicates(y_true, y_cl, matching)
    p12, losses = p1 & p2, []
    for a in matching:
        if not (p12[a].any() and p3[a].any()):
            continue  # No matched Louvain class for a, or no misses
        z_match, z_miss = z[p12[a]], z[p3[a]]  # both non empty
        z_match, z_miss = z_match.flatten(1), z_miss.flatten(1)
        index = NearestNeighbors(n_neighbors=min(k + 1, z_match.shape[0]))
        index.fit(_np(z_match))
        _, idx = index.kneighbors(_np(z_miss))
        idx = idx[:, 1:]  # exclude self as nearest neighbor
        if idx.size == 0:
            # Can happen if z_match or z_miss only have 1 row, which can happen
            # for clusters with only one element
            continue
        targets = z_match[torch.tensor(idx)].mean(dim=1)
        losses.append(torch.norm(z_miss - targets, dim=-1).mean())
    if not losses:
        return torch.tensor(0.0, requires_grad=True).to(z.device)
    return torch.stack(losses).mean() / sqrt(z.shape[-1])


def louvain_communities(
    z: np.ndarray | Tensor,
    k: int = 50,
    scaling: (
        Literal["standard", "minmax"] | TransformerMixin | None
    ) = "standard",
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

    Returns:
        Ok so this returns a lot of things:
        1. (`list[set[int]]`) The actual louvain communities, which is a
           partition of the set $\\\\{ 0, 1, ..., N-1 \\\\}$.
        2. (`np.ndarray`) The `(N,)` label vector for the communities. Let's
           call it `y_louvain`. If there are `c` communities, then `y_louvain`
           has integer values in $\\\\{ 0, 1, ..., c-1 \\\\}$, and if
           `y_louvain[i] == j`, then `z[i]` belongs to the `j`-th community
        4. (`np.ndarray`) The `(N, k)` distance matrix `m` of the rows of `z`
           to their `k` nearest neighbors: if $0 \\leq i < N$ and $0 \\leq j <
           k$, then `m[i, j] = np.linalg(z[i] - z[n])` where `z[n]` is the
           `j`-th nearest neighbor of `z[i]`.
        5. (`np.ndarray`) The `(N, k)` index matrix `u` of the nearest
           neighbors of the rows of `z`: if $0 \\leq i < N$ and $0 \\leq j <
           k$, then `u[i, j] = n` where `z[n]` is the `j`-th nearest neighbor
           of `z[i]`.
    """
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
        **({"backend": "cugraph"} if torch.cuda.is_available() else {}),  # type: ignore
    )
    y_louvain = [0] * len(graph)
    for i, c in enumerate(communities):
        for n in c:
            y_louvain[n] = i
    return communities, np.array(y_louvain)


def louvain_loss(z: Tensor, y_true: np.ndarray | Tensor, k: int = 5) -> Tensor:
    """
    Calls `nlnas.clustering.clustering_loss` with the Louvain clustering data.
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().detach().numpy()
    assert isinstance(y_true, np.ndarray)  # For typechecking
    _, y_louvain = louvain_communities(z)
    y_louvain = y_louvain[: len(y_true)]
    # TODO: Why is y_louvain sometimes longer than y_true?
    # This seems to only happen if z has nan's, in which case
    # len(y_louvain) = len(y_true) + 1
    matching = class_otm_matching(y_true, y_louvain)
    return clustering_loss(z, y_true, y_louvain, matching, k)


def otm_matching_predicates(
    y_a: np.ndarray,
    y_b: np.ndarray,
    matching: dict[int, set[int]] | dict[str, set[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Let `y_a` be `(N,)` integer array with values in $\\\\{ 0, 1, ..., c_a-1
    \\\\}$. If `y_a[i] == j`, then it is understood that the `i`-th sample (in
    some dataset, say `x`) is in class `j`, which for disambiguation we'll call
    the a-class `j`.

    Likewise, let `y_b` be `(N,)` integer array with values $\\\\{ 0, 1, ...,
    c_b-1 \\\\}$. If `y_b[i] == j`, then it is understood that the `i`-th
    sample `x[i]` is in b-class `j`.

    Finally, let `matching` be a (possibley one-to-many) matching between the
    a-classes and the b-classes. In other words each a-class corresponds to
    some set of b-classes.

    This method returns four boolean arrays with shape `(c_a, N)`, which in my
    head I call "true-louvain-miss-excess":

    1. `p1` is simply given by `p1[a] = (y_a == a)`, or in other words, `p1[a,
       i]` is `True` if and only if the `i`-th sample is in a-class `a`.
    2. `p2[a, i]` is `True` if and only if the `i`-th sample is in a b-class
       that has matched to a-class `a`.
    3. `p3` is (informally) given by `p3[a] = (p1[a] and not p2[a])`. In other
       words, `p3[a, i]` is `True` if sample `i` is in a-class `a` but not in
       any b-class matched with `a`.
    4. `p4` is the "dual" of `p3`: `p4[a] = (p2[a] and not p1[a])`. In other
       words, `p4[a, i]` is `True` if sample `i` is not in a-class `a`, but is
       in a b-class matched with `a`.

    I hope this all makes sense.

    Args:
        y_a (np.ndarray): A `(N,)` integer array with values in $\\\\{ 0, 1,
            ..., c_a - 1 \\\\}$ for some $c_a > 0$.
        y_b (np.ndarray): A `(N,)` integer array with values in $\\\\{ 0, 1,
            ..., c_b - 1 \\\\}$ for some $c_b > 0$.
        matching (dict[int, set[int]] | dict[str, set[int]]): A partition of
            $\\\\{ 0, ..., c_b - 1 \\\\}$ into $c_a$ sets. The $i$-th set is
            understood to be the set of all classes of `y_b` that matched with
            the $i$-th class of `y_a`. If some keys are strings, they must be
            convertible to ints.
    """
    m = {int(k): v for k, v in matching.items()}
    c_a = y_a.max() + 1
    p1 = [y_a == a for a in range(c_a)]
    p2 = [
        (
            np.sum([np.zeros_like(y_b)] + [y_b == b for b in m[a]], axis=0) > 0
            if a in m
            else np.full_like(y_a, False, dtype=bool)
        )  # a is not matched in m
        for a in range(c_a)
    ]
    p3 = [p1[a] & ~p2[a] for a in range(c_a)]
    p4 = [p2[a] & ~p1[a] for a in range(c_a)]
    return np.array(p1), np.array(p2), np.array(p3), np.array(p4)
