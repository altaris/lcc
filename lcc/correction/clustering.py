"""Clustering of latent representations"""

from itertools import product
from typing import Literal

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike

from ..utils import to_int_array
from .utils import Matching, to_int_matching


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
    and nodes of `set_b`. A one-to-many matching is like a usual matching except
    that a node in `set_a` can match to any number of nodes in `set_b` (whereas
    nodes in `set_b` can match at most one node in `set_a`).

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


def class_otm_matching(y_a: ArrayLike, y_b: ArrayLike) -> Matching:
    """
    Let `y_a` and `y_b` be `(N,)` integer arrays. We think of them as classes on
    some dataset, say `x`, which we call respectively *$a$-classes* and
    *$b$-classes*. This method performs a one-to-many matching from the classes
    in `y_a` to the classes in `y_b` to overall maximize the cardinality of the
    intersection between samples labeled by $a$ and matched $b$-classes.

    Example:

        >>> y_a = np.array([1, 1, 1, 1, 2, 2, 3, 4, 4])
        >>> y_b = np.array([10, 50, 10, 20, 20, 20, 30, 30, 30])
        >>> class_otm_matching(y_a, y_b)
        {1: {10, 50}, 2: {20}, 3: set(), 4: {30}}

        Here, `y_a` assigns class `1` to samples 0 to 3, label `2` to samples 4
        and 5 etc. On the other hand, `y_b` assigns its own classes to the
        dataset. What is the best way to regroup classes of `y_b` to
        approximate the labelling of `y_a`? The `class_otm_matching` return
        value argues that classes `10` and `15` should be regrouped under `1`
        (they fit neatly), label `20` should be renamed to `2` (eventhough it
        "leaks" a little, in that sample 3 is labelled with `1` and `20`), and
        class `30` should be renamed to `4`. No class in `y_b` is assigned to
        class `3` in this matching.

    Note:
        There are no restriction on the values of `y_a` and `y_b`. In
        particular, they need not be distinct: the following works fine

        >>> y_a = np.array([1, 1, 1, 1, 2, 2, 3, 4, 4])
        >>> y_b = np.array([1, 5, 1, 2, 2, 2, 3, 3, 3])
        >>> class_otm_matching(y_a, y_b)
        {1: {1, 5}, 2: {2}, 3: set(), 4: {3}}

    Warning:
        Negative labels in `y_a` or `y_b` are ignored. So the output matching
        dict will never have negative keys, and the sets will never have
        negative values either.

    Args:
        y_a (ArrayLike): A `(N,)` integer array.
        y_b (ArrayLike): A `(N,)` integer array.

    Returns:
        A dict that maps each class in `y_a` to the set of classes in `y_b` that
        it has matched.
    """
    y_a, y_b, match_graph = to_int_array(y_a), to_int_array(y_b), nx.DiGraph()
    for i, j in product(np.unique(y_a), np.unique(y_b)):
        if i < 0 or j < 0:
            continue
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


def otm_matching_predicates(
    y_a: ArrayLike,
    y_b: ArrayLike,
    matching: Matching,
    c_a: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Let `y_a` be `(N,)` integer array with values in $\\\\{ 0, 1, ..., c_a - 1
    \\\\}$ (if the argument `c_a` is `None`, it is inferred to be `y_a.max() +
    1`). If `y_a[i] == j`, then it is understood that the $i$-th sample (in
    some dataset, say `x`) is in class $j$, which for disambiguation we'll call
    the $a$-class $j$.

    Likewise, let `y_b` be `(N,)` integer array with values $\\\\{ 0, 1, ...,
    c_b - 1 \\\\}$. If `y_b[i] == j`, then it is understood that the $i$-th
    sample `x[i]` is in $b$-class $j$.

    Finally, let `matching` be a (possibley one-to-many) matching between the
    $a$-classes and the $b$-classes. In other words each $a$-class corresponds to
    some set of $b$-classes.

    This method returns four boolean arrays with shape `(c_a, N)`, which in my
    head I call *"true-louvain-miss-excess"*:

    1. `p1` is simply given by `p1[a] = (y_a == a)`, or in other words, `p1[a,
       i]` is `True` if and only if the $i$-th sample is in $a$-class `a`.
    2. `p2[a, i]` is `True` if and only if the $i$-th sample is in a $b$-class
       that has matched to $a$-class `a`.
    3. `p3` is (informally) given by `p3[a] = (p1[a] and not p2[a])`. In other
       words, `p3[a, i]` is `True` if sample $i$ is in $a$-class `a` but not in
       any $b$-class matched with `a`.
    4. `p4` is the "dual" of `p3`: `p4[a] = (p2[a] and not p1[a])`. In other
       words, `p4[a, i]` is `True` if sample $i$ is not in $a$-class `a`, but is
       in a $b$-class matched with `a`.

    I hope this all makes sense.

    Args:
        y_a (np.ndarray): A `(N,)` integer array with values in $\\\\{ 0, 1,
            ..., c_a - 1 \\\\}$ for some $c_a > 0$.
        y_b (np.ndarray): A `(N,)` integer array with values in $\\\\{ 0, 1,
            ..., c_b - 1 \\\\}$ for some $c_b > 0$.
        matching (Matching): A partition of
            $\\\\{ 0, ..., c_b - 1 \\\\}$ into $c_a$ sets. The $i$-th set is
            understood to be the set of all classes of `y_b` that matched with
            the $i$-th class of `y_a`. If some keys are strings, they must be
            convertible to ints. This has probably been produced by
            `lcc.correction.class_otm_matching`.
        c_a (int | None, optional): Number of $a$-classes. Useful if `y_a`
            does not contain all the possible classes of the dataset at hand.
            If `None`, then `y_a` is assumed to contain all classes, and so `c_a
            = y_a.max() + 1`.
    """
    y_a, y_b = to_int_array(y_a), to_int_array(y_b)
    matching = to_int_matching(matching)
    if (la := len(y_a)) != (lb := len(y_b)):
        raise ValueError(
            f"y_a and y_b must have the same length, got {la} and {lb}"
        )
    c_a = c_a or int(y_a.max() + 1)
    p1 = [y_a == a for a in range(c_a)]
    p2 = [
        (
            np.sum(
                [np.zeros_like(y_b)] + [y_b == b for b in matching.get(a, [])],
                axis=0,
            )
            > 0
            if a in matching
            else np.full_like(y_b, False, dtype=bool)  # a isn't matched in m
        )
        for a in range(c_a)
    ]
    p3 = [p1[a] & ~p2[a] for a in range(c_a)]
    p4 = [p2[a] & ~p1[a] for a in range(c_a)]
    return np.array(p1), np.array(p2), np.array(p3), np.array(p4)
