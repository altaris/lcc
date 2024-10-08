from datetime import datetime

import cuml
import faiss
import faiss.contrib.torch_utils
import networkx as nx
import numpy as np
import torch
from loguru import logger as logging
from tqdm import tqdm

from nlnas.classifiers import HuggingFaceClassifier
from nlnas.datasets import HuggingFaceDataset
from nlnas.datasets.utils import dl_head

BATCH_SIZE = 1024

logging.info("Loading dataset")
dataset = HuggingFaceDataset(
    "cifar100",
    fit_split="train[:80%]",
    val_split="train[80%:]",
    test_split="test",
    image_processor=HuggingFaceClassifier.get_image_processor(
        "microsoft/resnet-18"
    ),
    train_dl_kwargs={"batch_size": BATCH_SIZE, "drop_last": True},
)
IMAGE_KEY = "img"
# dataset = HuggingFaceDataset(
#     "ILSVRC/imagenet-1k",
#     fit_split="train",
#     val_split="validation",
#     test_split="test",
#     image_processor=HuggingFaceClassifier.get_image_processor(
#         "microsoft/resnet-18"
#     ),
#     train_dl_kwargs={"batch_size": BATCH_SIZE, "drop_last": True},
# )
# IMAGE_KEY = "image"
dataset.setup("fit")

dl = dataset.train_dataloader()
z = dl_head(dl, 1)[0][IMAGE_KEY]
z = z.flatten(1)
n_features = z.shape[-1]
logging.debug("Feature dimension: {}", n_features)

PCA_N_COMPONENTS = 1024
pca = cuml.IncrementalPCA(n_components=PCA_N_COMPONENTS)
logging.info("Fitting PCA")
start = datetime.now()
dl = dataset.train_dataloader()
for batch in tqdm(dl):
    x = batch[IMAGE_KEY].flatten(1)
    x = x.to("cuda")
    pca.partial_fit(x)
logging.info("PCA fitted in {}", datetime.now() - start)

logging.info("Building index")
index = faiss.index_cpu_to_gpu(
    faiss.StandardGpuResources(),
    0,
    faiss.IndexFlatL2(PCA_N_COMPONENTS),
)
start = datetime.now()
dl = dataset.train_dataloader()
for batch in tqdm(dl):
    x = batch[IMAGE_KEY].flatten(1)
    x = x.to("cuda")
    x = pca.transform(x)
    x = torch.as_tensor(x, device="cuda").float()
    index.add(x)
logging.info("Index built in {}", datetime.now() - start)

logging.info("Construting KNN graph")
start = datetime.now()
graph, n = nx.DiGraph(), 0
dl = dataset.train_dataloader()
for batch in tqdm(dl):
    x = batch[IMAGE_KEY].flatten(1)
    x = x.to("cuda")
    x = pca.transform(x)
    x = torch.as_tensor(x, device="cuda").float()
    dst, idx = index.search(x, 5)
    dst, idx = dst.cpu().numpy(), idx.cpu().numpy()
    for j, all_i, all_d in zip(range(len(idx)), idx, dst):
        graph.add_weighted_edges_from(
            [
                (n + j, int(i), np.exp(-d / np.sqrt(n_features)))
                for i, d in zip(all_i, all_d)
            ],
            weight="weight",
        )
    n += len(x)
# Reciprocal edges share the same weight, so it's ok to discard one of them
# See https://networkx.org/documentation/stable/reference/classes/generated/networkx.DiGraph.to_undirected.html
graph = nx.to_undirected(graph)
logging.info("Constructed KNN graph in {}", datetime.now() - start)
logging.debug(
    "Graph has {} nodes and {} edges",
    graph.number_of_nodes(),
    graph.number_of_edges(),
)

logging.info("Finding Louvain communities")
start = datetime.now()
# communities: list[set[int]] = cugraph.leiden(graph)
communities: list[set[int]] = nx.community.louvain_communities(
    graph, backend="cugraph"
)
logging.info("Found Louvain communities in {}", datetime.now() - start)
logging.debug("Found {} communities", len(communities))
logging.debug(
    "Community size: min={}, max={}, med={}",
    min(len(c) for c in communities),
    max(len(c) for c in communities),
    np.median([len(c) for c in communities]),
)
