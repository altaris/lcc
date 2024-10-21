"""Clustering of latent representations"""

from collections import defaultdict
from itertools import product
from math import sqrt
from typing import Literal

import networkx as nx
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..utils import make_tqdm, to_array, to_tensor


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


def _mc_cc_predicates(
    y_true: np.ndarray | Tensor | list[int],
    y_clst: np.ndarray | Tensor | list[int],
    matching: dict[int, set[int]] | dict[str, set[int]],
    n_true_classes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns two boolean arrays (also called predicates) `p_mc` and `p_cc` (in
    this order), both of shape `(n_true_classes, N)`, where:
    - `p_mc[i_true, j]` is `True` if the $j$-th sample is in true class `i_true`
      and misclustered (i.e. not in any cluster matched with true class
      `i_true`);
    - `p_cc[i_true, j]` is `True` if the $j$-th sample is in true class `i_true`
      and correctly clustered (i.e. in a cluster matched with true class
      `i_true`).

    Note:
        `p_mc != ~p_cc` in general ;)

    Args:
        y_true (np.ndarray | Tensor | list[int]): A `(N,)` integer array.
        y_clst (np.ndarray | Tensor | list[int]): A `(N,)` integer array.
        matching (dict[int, set[int]] | dict[str, set[int]]):
        n_true_classes (int | None, optional): Number of true classes. Useful if
            `y_true` is a slice of the real true label vector and does not
            contain all the possible true classes of the dataset at hand.  If
            `None`, then `y_true` is assumed to contain all classes, and so
            `n_true_classes` defaults to `y_true.max() + 1`.
    """
    y_true, y_clst = to_array(y_true), to_array(y_clst)
    matching = {int(a): bs for a, bs in matching.items()}
    p1, p2, p_mc, _ = otm_matching_predicates(
        y_true, y_clst, matching, c_a=n_true_classes or int(y_true.max() + 1)
    )
    return p_mc, p1 & p2


def class_otm_matching(
    y_a: np.ndarray | Tensor | list[int], y_b: np.ndarray | Tensor | list[int]
) -> dict[int, set[int]]:
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
        y_a (np.ndarray | Tensor | list[int]): A `(N,)` integer array.
        y_b (np.ndarray | Tensor | list[int]): A `(N,)` integer array.

    Returns:
        A dict that maps each class in `y_a` to the set of classes in `y_b` that
        it has matched.
    """
    y_a, y_b, match_graph = to_array(y_a), to_array(y_b), nx.DiGraph()
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


# def lcc_knn_indices(
#     dl: EMETDDataLoader,
#     y_true: np.ndarray | Tensor | list[int],
#     y_clst: np.ndarray | Tensor | list[int],
#     matching: dict[int, set[int]] | dict[str, set[int]],
#     k: int,
#     n_true_classes: int | None = None,
#     device: Any = None,
#     tqdm_style: Literal["notebook", "console", "none"] | None = None,
# ) -> dict[int, faiss.IndexHNSWFlat]:
#     """
#     The matching between true classes and cluster classes reveal which samples
#     are correctly clustered and which aren't. This method fits some KNN indices
#     on correctly clustered samples in a given class.

#     Say the underlying dataset has `N` samples with `d` dimensions (aka
#     features). Then this methods returns a dict where
#     - the keys are *among* true classes (unique values of `y_true`);
#     - if `i_true` is a true class in the dict, then the value at key `i_true` is
#       a faiss KNN index fitted on the currectly clustered samples of true class
#       `i_true`.

#     Warning:
#         Not every true class might be represented in the return dict. For
#         example, in the ideal scenario where `y_true == y_clst` (up to label
#         permutation), the returned dict would be empty. More generally, if a
#         class has less than `k` misclustered samples, then it is not included.

#     TODO:
#         Allow `dl` to also be just a regular `DataLoader`.

#     Args:
#         dl (EMETDDataLoader): A dataloader over a tensor dataset.
#         y_true (np.ndarray | Tensor | list[int]): A `(N,)` integer array.
#         y_clst (np.ndarray | Tensor | list[int]): A `(N,)` integer array.
#         matching (dict[int, set[int]] | dict[str, set[int]]): Produced by
#             `nlnas.correction.class_otm_matching`.
#         k (int, optional):
#         n_true_classes (int | None, optional): Number of true classes. Useful if
#             `y_true` is a slice of the real true label vector and does not
#             contain all the possible true classes of the dataset at hand. If
#             `None`, then `y_true` is assumed to contain all classes, and so
#             `n_true_classes` defaults to `y_true.max() + 1`.
#         device (Any, optional): If left to `None`,
#             uses CUDA if it is available, otherwise falls back to CPU. Setting
#             `cuda` while CUDA isn't available will **silently** fall back to
#             CPU.
#     """

#     def _batches(desc: str | None = None) -> Iterator[Tensor]:  # shorthand
#         xs = make_tqdm(tqdm_style)(dl, desc, leave=False) if desc else dl
#         for x in xs:
#             yield x.flatten(1).to(device)

#     z = next(_batches())
#     z = z.flatten(1)
#     n_features = z.shape[-1]

#     result = {}
#     p_mc, p_cc = _mc_cc_predicates(
#         y_true, y_clst, matching, n_true_classes=n_true_classes
#     )
#     for i_true, (p_cc_i, p_mc_i) in enumerate(
#         make_tqdm(tqdm_style)(
#             zip(p_cc, p_mc), "Fitting CC KNN indices", leave=False
#         )
#     ):
#         if p_cc_i.sum() < k:  # Not enough corr. clst. samples in this class
#             continue
#         if not p_mc_i.any():  # No missclst. samples in this class
#             continue
#         index = faiss.IndexHNSWFlat(n_features, k)
#         for z in make_tqdm(tqdm_style)(
#             dl.add_mask(p_cc_i),
#             f"Fitting CC KNN index for y_true={i_true}",
#             leave=False,
#         ):
#             index.add(to_array(z).astype(np.float32))
#         result[i_true] = index
#     return result


def lcc_loss(
    z: Tensor,
    y_true: np.ndarray | Tensor | list[int],
    y_clst: np.ndarray | Tensor | list[int],
    matching: dict[int, set[int]] | dict[str, set[int]],
    targets: dict[int, Tensor],
    n_true_classes: int | None = None,
) -> Tensor:
    """
    Derives the clustering correction loss from a tensor of latent
    representation `z` and dict of targets (see `nlnas.correction.lcc_targets`).

    First, recall that the values of `target` (as produced
    `nlnas.correction.lcc_targets`) are `(k, d)` tensors, for some length `k`.

    Let's say `a` is a missclusterd latent sample (a.k.a. a row of `z`) in true
    class `i_true`, and that `(b_1, ..., b_k)` are the rows of
    `targets[i_true]`. Then `a` contributes a term to the LCC loss equal to the
    distance between `a` and the closest `b_j`, divided by $\\\\sqrt{d}$.

    It is possible that `i_true` is not in the keys of `targets`, in which case
    the contribution of `a` to the LCC loss is zero. In particular, if `targets`
    is empty, then the LCC loss is zero.

    Usage:

        ```python
        targets = lcc_targets(
            dl,
            y_true,
            y_clst,
            matching,
            n_true_classes=n_true_classes,
        )
        loss = lcc_loss(
            z,
            y_true,
            y_clst,
            matching,
            targets,
            n_true_classes=n_true_classes,
        )
        ```

    Args:
        z (Tensor): The tensor of latent representations. *Do not* mask it
            before passing it to this method.  The correctly samples and the
            missclustered samples are automatically separated.
        y_true (np.ndarray | Tensor | list[int]): A `(N,)` integer array of true
            labels.
        y_clst (np.ndarray | Tensor | list[int]): A `(N,)` integer array of the
            cluster labels.
        matching (dict[int, set[int]] | dict[str, set[int]]): As produced by
            `nlnas.correction.class_otm_matching`.
        targets (dict[int, Tensor]): As produced by
            `nlnas.correction.lcc_targets`.
        n_true_classes (int | None, optional): Number of true classes. Useful if
            `y_true` is a slice of the actual true label vector and does not
            contain all the possible true classes.
    """
    z = z.flatten(1)
    if not targets:
        # ↓ actually need grad?
        return torch.tensor(0.0, requires_grad=True).to(z.device)
    y_true = to_tensor(y_true)
    p_mc, _ = _mc_cc_predicates(
        y_true, y_clst, matching, n_true_classes=n_true_classes
    )
    sqrt_d, losses = sqrt(z.shape[-1]), []
    for i_true, p_mc_i_true in enumerate(p_mc):
        if not (i_true in targets and len(targets[i_true]) > 0):  # no targets
            continue
        if not p_mc_i_true.any():  # every sample is correctly clustered
            continue
        d = torch.cdist(z[p_mc_i_true], targets[i_true]) / sqrt_d
        losses.append(d.min(dim=-1).values)
    if not losses:
        return torch.tensor(0.0, requires_grad=True).to(z.device)
    return torch.concat(losses).mean()


def lcc_targets(
    dl: DataLoader,
    y_true: np.ndarray | Tensor | list[int],
    y_clst: np.ndarray | Tensor | list[int],
    matching: dict[int, set[int]] | dict[str, set[int]],
    n_true_classes: int | None = None,
    tqdm_style: Literal["notebook", "console", "none"] | None = None,
) -> dict[int, Tensor]:
    """
    Provides the correction targets for misclustered samples in `z`. In more
    details, this method returns a dict where:
    - the keys are *among* true classes (unique values of `y_true`) and in fact
      are the same keys as `knn_indices`; let's say that `i_true` is a key that
      owns `k` clusters;
    - the associated value a `(k, d)` tensor, where `d` is the latent dimension,
      whose rows, which correspond to clusters matched to `i_true`, is a random
      correctly clustered sample in that cluster.

    Under the hood, this method first choose the samples by their index based on
    the "correctly clustered" predicate of `_mc_cc_predicates`. Then, the whole
    dataset is iterated to collect the actual samples.

    Args:
        dl (DataLoader): A dataloader over a tensor dataset.
        y_true (np.ndarray | Tensor): A `(N,)` integer array.
        y_clst (np.ndarray | Tensor): A `(N,)` integer array.
        matching (dict[int, set[int]] | dict[str, set[int]]): Produced by
            `nlnas.correction.class_otm_matching`.
        knn_indices (dict[int, tuple[Any, Tensor]]): As produced by
            `lcc_knn_indices`
        n_true_classes (int | None, optional): Number of true classes. Useful if
            `y_true` is a slice of the real true label vector and does not
            contain all the possible true classes of the dataset at hand. If
            `None`, then `y_true` is assumed to contain all classes, and so
            `n_true_classes` defaults to `y_true.max() + 1`.
        tqdm_style (Literal["notebook", "console", "none"] | None, optional):
    """
    matching = {int(k): v for k, v in matching.items()}
    _, p_cc = _mc_cc_predicates(y_true, y_clst, matching, n_true_classes)
    indices: dict[int, list[int]] = defaultdict(list)
    for i_true, p_cc_i_true in enumerate(p_cc):
        for j_clst in matching[i_true]:
            p = p_cc_i_true & (y_clst == j_clst)
            indices[i_true].append(np.random.choice(np.where(p)[0]))
    n_seen, n_todo = 0, sum(len(v) for v in indices.values())
    result: dict[int, list[Tensor]] = defaultdict(list)
    for batch in make_tqdm(tqdm_style)(
        dl, "Finding correction targets", leave=False
    ):
        for i_true, idxs in indices.items():
            lst = [idx for idx in idxs if n_seen <= idx < n_seen + len(batch)]
            for idx in lst:
                result[i_true].append(batch[idx - n_seen])
                n_todo -= 1
        if n_todo <= 0:
            return {k: torch.stack(v).flatten(1) for k, v in result.items()}
        n_seen += len(batch)
    raise RuntimeError("Some correction targets could not be found")


def otm_matching_predicates(
    y_a: np.ndarray | Tensor | list[int],
    y_b: np.ndarray | Tensor | list[int],
    matching: dict[int, set[int]] | dict[str, set[int]],
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
        matching (dict[int, set[int]] | dict[str, set[int]]): A partition of
            $\\\\{ 0, ..., c_b - 1 \\\\}$ into $c_a$ sets. The $i$-th set is
            understood to be the set of all classes of `y_b` that matched with
            the $i$-th class of `y_a`. If some keys are strings, they must be
            convertible to ints. This has probably been produced by
            `nlnas.correction.class_otm_matching`.
        c_a (int | None, optional): Number of $a$-classes. Useful if `y_a`
            does not contain all the possible classes of the dataset at hand.
            If `None`, then `y_a` is assumed to contain all classes, and so `c_a
            = y_a.max() + 1`.
    """
    if (la := len(y_a)) != (lb := len(y_b)):
        raise ValueError(
            f"y_a and y_b must have the same length, got {la} and {lb}"
        )
    y_a, y_b = to_array(y_a), to_array(y_b)
    c_a = c_a or int(y_a.max() + 1)
    m = {int(k): v for k, v in matching.items()}
    p1 = [y_a == a for a in range(c_a)]
    p2 = [
        (
            np.sum(
                [np.zeros_like(y_b)] + [y_b == b for b in m.get(a, [])],
                axis=0,
            )
            > 0
            if a in m
            else np.full_like(y_b, False, dtype=bool)  # a isn't matched in m
        )
        for a in range(c_a)
    ]
    p3 = [p1[a] & ~p2[a] for a in range(c_a)]
    p4 = [p2[a] & ~p1[a] for a in range(c_a)]
    return np.array(p1), np.array(p2), np.array(p3), np.array(p4)
