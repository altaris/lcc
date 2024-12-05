import os

import networkx as nx
import numpy as np
import torch
from faiss import IndexHNSWFlat
from lightning_fabric import Fabric
from loguru import logger as logging

# from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from lcc.classifiers import HuggingFaceClassifier
from lcc.datasets import HuggingFaceDataset
from lcc.datasets.wrapped import DEFAULT_DATALOADER_KWARGS
from lcc.utils import to_array

N_DEVICES = os.getenv("N_DEVICES", 1)
K = 10
LATENT_DIM_OVERRIDE = 2000  # Truncate tensors for quicker testing

HF_MODEL_NAME = "microsoft/resnet-18"

DATASET = "cifar100"
TRAIN_SPLIT = "train"
VAL_SPLIT = "train[80%:]"
TEST_SPLIT = "test"
IMAGE_KEY = "img"
LABEL_KEY = "fine_label"


# def all_gather_concat(fabric: Fabric, x: Tensor) -> Tensor:
#     """
#     `all_gather`s and concatenate 2D tensors across all ranks. `x` is expected
#     to have shape `(n, d)`, where `d` is the same across ranks, but where `n`
#     can vary (as opposed to `Fabric.all_gather`, where all tensors are truncated
#     to match the length of the shortest one).
#     """
#     lengths = fabric.all_gather(torch.tensor(len(x)))
#     assert isinstance(lengths, torch.Tensor)
#     max_lengths = int(lengths.max())
#     x = torch.nn.functional.pad(x, (0, 0, 0, max_lengths - len(x)))
#     all_x = fabric.all_gather(x)
#     x = torch.cat([t[:s] for t, s in zip(all_x, lengths) if s > 0])
#     assert len(x) == int(lengths.sum())  # FOR DEBUGGING
#     return x


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    fabric = Fabric(devices=N_DEVICES)
    fabric.launch()
    logging.info("[RANK {}] Started", fabric.global_rank)

    image_processor = HuggingFaceClassifier.get_image_processor(HF_MODEL_NAME)
    ds = HuggingFaceDataset(
        dataset_name=DATASET,
        fit_split=TRAIN_SPLIT,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT,
        label_key=LABEL_KEY,
        image_processor=image_processor,
    )
    ds.setup("fit")

    # latent_dim = next(iter(dl))[IMAGE_KEY].flatten(1).shape[-1]
    # r0_debug("Latent dimension: {}", latent_dim)

    # dl = fabric.setup_dataloaders(ds.train_dataloader())

    if fabric.world_size > 1:
        ds = ds._get_dataset("train")
        sampler = DistributedSampler(ds, shuffle=False)
        dl = DataLoader(
            ds,
            batch_size=DEFAULT_DATALOADER_KWARGS["batch_size"],
            num_workers=1,
            shuffle=False,
            sampler=sampler,
        )
    else:
        dl = ds.train_dataloader()

    index = IndexHNSWFlat(LATENT_DIM_OVERRIDE, K + 1)
    _absolute_indices: list[torch.Tensor] = []
    desc = f"[RANK {fabric.global_rank}] Building KNN index (k={K})"
    for batch in tqdm(dl, desc, leave=False):
        _absolute_indices.append(batch["_idx"])
        z = batch[IMAGE_KEY].flatten(1)[:, :LATENT_DIM_OVERRIDE]
        z = to_array(z).astype(np.float32)
        index.add(z)
    absolute_indices = torch.cat(_absolute_indices, dim=0)
    logging.info(
        "[RANK {}] Finished building KNN index (n_smpl={})",
        fabric.global_rank,
        index.ntotal,
    )

    dl = ds.train_dataloader()  # Full dl
    desc = f"[RANK {fabric.global_rank}] Querying KNN index (k={K})"
    graph = nx.Graph()
    for batch in tqdm(dl, desc, leave=False):
        z = batch[IMAGE_KEY].flatten(1)[:, :LATENT_DIM_OVERRIDE]
        z = to_array(z).astype(np.float32)
        dst, idx_knn = index.search(z, K + 1)
        idx_knn = absolute_indices[idx_knn]  # (bs, K+1)

        dst, idx_knn = fabric.all_gather((dst, idx_knn))
        if fabric.world_size > 1:
            dst = dst.permute(1, 0, 2).flatten(1)
            idx_knn = idx_knn.permute(1, 0, 2).flatten(1)
        arso = dst.argsort(dim=1, descending=False)
        arso = arso[:, 1 : K + 1]  # Ignore self as KNN
        r = torch.arange(len(z)).unsqueeze(1).expand(len(z), K)
        # r: (bs, K) = [[0, ..., 0], ..., [bs-1, ..., bs-1]]
        idx_knn = idx_knn[r, arso]

        idx = batch["_idx"].to(idx_knn)  # (bs,)
        idx = idx.repeat(K, 1).T  # (bs, K) I swear it's correct
        e = torch.stack(
            [idx.flatten(), idx_knn.flatten()], dim=-1
        )  # (bs * (K+1), 2)  I swear this is also correct
        graph.add_edges_from(e.cpu().numpy())

    graph.remove_edges_from(nx.selfloop_edges(graph))  # Just to be sure

    logging.info(
        "[RANK {}] Finished constructing KNN graph: {} nodes, {} edges",
        fabric.global_rank,
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )

    if fabric.global_rank == 0:
        path = "./graph-cifar100.sparse6"
        logging.info("[RANK {}] Saving graph to {}", fabric.global_rank, path)
        nx.write_sparse6(graph, path)
