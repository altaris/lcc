"""Everything related to LCC."""

from typing import Literal

import numpy as np
import torch

from ..utils import to_array
from .choice import (
    LCC_CLASS_SELECTIONS,
    GraphTotallyDisconnected,
    LCCClassSelection,
    choose_classes,
    confusion_graph,
    heaviest_connected_subgraph,
    max_connected_confusion_choice,
    top_confusion_pairs,
)
from .clustering import (
    CLUSTERING_METHODS,
    ClusteringMethod,
    class_otm_matching,
    lcc_knn_indices,
    lcc_loss,
    lcc_targets,
    otm_matching_predicates,
)
from .louvain import louvain_communities

__all__ = [
    "choose_classes",
    "class_otm_matching",
    "CLUSTERING_METHODS",
    "ClusteringMethod",
    "confusion_graph",
    "get_cluster_labels",
    "GraphTotallyDisconnected",
    "heaviest_connected_subgraph",
    "LCC_CLASS_SELECTIONS",
    "lcc_knn_indices",
    "lcc_loss",
    "lcc_targets",
    "LCCClassSelection",
    "louvain_communities",
    "max_connected_confusion_choice",
    "otm_matching_predicates",
    "top_confusion_pairs",
]


# pylint: disable=duplicate-code
def get_cluster_labels(
    z: np.ndarray | torch.Tensor | list[float],
    method: ClusteringMethod = "louvain",
    scaling: Literal["standard", "minmax"] | None = "standard",
    device: Literal["cpu", "cuda"] | None = None,
    **kwargs,
) -> np.ndarray:
    """
    Convenience method that dispatches to the appropriate clustering algorithm.
    Also performs some preprocessing as required.

    Args:
        z (np.ndarray | Tensor): The latent representation tensor.
        method (Literal["louvain", "dbscan", "hdbscan"], optional): Defaults to
            `"louvain"`. See `nlnas.correction.CLUSTERING_METHODS`.
        scaling (Literal["standard", "minmax"] | None, optional): Defaults to
            `"standard"`.
        device (Literal["cpu", "cuda"] | None, optional): If left to `None`,
            uses CUDA if it is available, otherwise falls back to CPU. Setting
            `cuda` while CUDA isn't available will **silently** fall back to
            CPU.
        **kwargs: Passed to the clustering object.
    """
    use_cuda = (
        device == "cuda" or device is None
    ) and torch.cuda.is_available()

    if use_cuda:
        from cuml.cluster import DBSCAN, HDBSCAN
        from cuml.preprocessing import MinMaxScaler, StandardScaler
    else:
        from sklearn.cluster import DBSCAN, HDBSCAN
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

    z = to_array(z)
    z = z.reshape(len(z), -1)
    if scaling == "standard":
        z = StandardScaler().fit_transform(z)
    elif scaling == "minmax":
        z = MinMaxScaler().fit_transform(z)
    if method == "louvain":
        return louvain_communities(z, device=device, **kwargs)[1]
    if method == "dbscan":
        return DBSCAN(**kwargs).fit(z).labels_
    if method == "hdbscan":
        return HDBSCAN(**kwargs).fit(z).labels_
    raise ValueError(f"Unsupported clustering method '{method}'")
