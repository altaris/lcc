from datetime import datetime
from typing import Literal

import networkx as nx
import numpy as np
import torch
from cuml.neighbors import NearestNeighbors as CumlNearestNeighbors
from sklearn.base import TransformerMixin
from sklearn.neighbors import NearestNeighbors as SklearnNearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import Tensor


def louvain_communities(
    nearest_neighbors_cls,
    z: np.ndarray | Tensor,
    k: int = 50,
    scaling: Literal["standard", "minmax"]
    | TransformerMixin
    | None = "standard",
) -> tuple[list[set[int]], np.ndarray]:
    if scaling == "standard":
        scaling = StandardScaler()
    elif scaling == "minmax":
        scaling = MinMaxScaler()
    if isinstance(z, Tensor):
        z = z.cpu().detach().numpy()

    print("start")

    z = z.reshape(len(z), -1)
    z = z if scaling is None else scaling.fit_transform(z)  # type: ignore
    assert isinstance(z, np.ndarray)  # for typechecking

    print("scaled")

    index = nearest_neighbors_cls(n_neighbors=min(k + 1, z.shape[0]))
    index.fit(z)
    adj = index.kneighbors_graph(z)

    print("kn graph adj")

    graph = nx.from_scipy_sparse_array(adj, edge_attribute="weight")
    graph.remove_edges_from(nx.selfloop_edges(graph))  # exclude self as NN
    # graph = nxcu.from_networkx(graph)

    print("kn graph")

    communities: list[set[int]] = nx.community.louvain_communities(  # type: ignore
        graph,
        **({"backend": "cugraph"} if torch.cuda.is_available() else {}),  # type: ignore
    )

    print("coms")

    y_louvain = [0] * len(graph)
    for i, c in enumerate(communities):
        for n in c:
            y_louvain[n] = i

    print("y_louvain")

    return communities, np.array(y_louvain)


n, d, k = 10000, 4096, 5
x = np.random.random((n, d))


start = datetime.now()
louvain_communities(SklearnNearestNeighbors, x, k)
print("SKLEARN:", datetime.now() - start)

start = datetime.now()
louvain_communities(CumlNearestNeighbors, x, k)
print("CUML:", datetime.now() - start)
