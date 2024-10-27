"""
This module revolves around the implementation of
`full_dataset_latent_clustering`.
"""

from pathlib import Path
from typing import Literal
from uuid import uuid4

import torch
import torch.distributed
from safetensors import torch as st
from torch import Tensor
from torch.utils.data import DataLoader

from nlnas.correction.clustering import otm_matching_predicates

from ..correction import (
    class_otm_matching,
    lcc_targets,
    louvain_communities,
)
from ..datasets import BatchedTensorDataset
from ..utils import (
    make_tqdm,
)
from .base import BaseClassifier, LatentClusteringData


def _fde_pca(
    model: BaseClassifier,
    dl: DataLoader,
    output_dir: str | Path,
    pca_dim: dict[str, int | None],
    tqdm_style: Literal["notebook", "console", "none"] | None = None,
) -> None:
    """
    `full_dataset_evaluation` in the case where PCA dim-redux has to be applied.
    See `full_dataset_evaluation` for the precise meaning of the arguments. Note
    that in this case, `pca_dim` mist be a dict that maps LCC submodule names to
    PCA dimensions.
    """
    if model.device.type == "cuda":
        from cuml import IncrementalPCA

        pcas = {
            sm: IncrementalPCA(n_components=d, output_type="numpy")
            for sm, d in pca_dim.items()
            if d
        }
    else:
        from sklearn.decomposition import IncrementalPCA

        pcas = {
            sm: IncrementalPCA(n_components=d)
            for sm, d in pca_dim.items()
            if d
        }

    rank = model.trainer.global_rank
    output_dir, tqdm = Path(output_dir), make_tqdm(tqdm_style)
    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        model.eval()
        for batch in tqdm(dl, f"[Rank {rank}] Fitting PCA(s)", leave=False):
            data: dict[str, Tensor] = {}
            model.forward_intermediate(
                inputs=batch[model.hparams["image_key"]],
                submodules=model.lcc_submodules,
                output_dict=data,
                keep_gradients=False,
            )
            for sm, z in data.items():
                if sm in pcas:
                    pcas[sm].partial_fit(z.flatten(1))
        for batch in tqdm(dl, f"[Rank {rank}] Evaluating", leave=False):
            data = {}
            y_pred = model.forward_intermediate(
                inputs=batch[model.hparams["image_key"]],
                submodules=model.lcc_submodules,
                output_dict=data,
                keep_gradients=False,
            )
            assert isinstance(y_pred, Tensor)
            data["y_true"] = batch[model.hparams["label_key"]]
            data["y_pred"] = y_pred
            for sm, z in data.items():
                if sm in pcas:
                    z = pcas[sm].transform(z.flatten(1))
                bid = uuid4().hex
                st.save_file(
                    {
                        "": z.flatten(1) if z.ndim > 1 else z,
                        "_idx": batch["_idx"],
                    },
                    output_dir / f"{sm}.{bid}.st",
                )
        model.train()


def _fde_no_pca(
    model: BaseClassifier,
    dl: DataLoader,
    output_dir: str | Path,
    tqdm_style: Literal["notebook", "console", "none"] | None = None,
) -> None:
    """
    `full_dataset_evaluation` in the case where no PCA dim-redux is to be
    applied. See `full_dataset_evaluation` for the precise meaning of the
    arguments.
    """
    rank = model.trainer.global_rank
    output_dir, tqdm = Path(output_dir), make_tqdm(tqdm_style)
    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        model.eval()
        for batch in tqdm(dl, f"[Rank {rank}] Evaluating", leave=False):
            data: dict[str, Tensor] = {}
            y_pred = model.forward_intermediate(
                inputs=batch[model.hparams["image_key"]],
                submodules=model.lcc_submodules,
                output_dict=data,
                keep_gradients=False,
            )
            assert isinstance(y_pred, Tensor)
            data["y_true"] = batch[model.hparams["label_key"]]
            data["y_pred"] = y_pred
            for sm, z in data.items():
                bid = uuid4().hex
                st.save_file(
                    {
                        "": z.flatten(1) if z.ndim > 1 else z,
                        "_idx": batch["_idx"],
                    },
                    output_dir / f"{sm}.{bid}.st",
                )
        model.train()


def _fdlc_r0(
    model: BaseClassifier,
    output_dir: str | Path,
    tqdm_style: Literal["notebook", "console", "none"] | None = None,
) -> dict[str, LatentClusteringData]:
    """
    Full dataset latent clustering, to only be executed on rank 0. It is assumed
    that the whole dataset has already been evaluated.
    """
    y_true, idx = BatchedTensorDataset(output_dir, prefix="y_true").load()
    y_true = y_true[idx.argsort()]  # y_true is not in order (of the dataset)
    n_classes = len(torch.unique(y_true))

    lcc_data: dict[str, LatentClusteringData] = {}
    lcc_kwargs = model.hparams.get("lcc_kwargs", {})

    tqdm = make_tqdm(tqdm_style)
    for sm in tqdm(model.lcc_submodules, "Clustering", leave=False):
        ds, idx = BatchedTensorDataset(output_dir, prefix=sm).extract_idx()
        dl = DataLoader(ds, batch_size=256)
        _, y_clst = louvain_communities(
            dl,
            k=lcc_kwargs.get("k", 5),
            device=model.device,
            tqdm_style=tqdm_style,
        )
        _y_true = y_true[idx]  # Match the order of y_clst
        matching = class_otm_matching(_y_true, y_clst)
        _, _, p, _ = otm_matching_predicates(_y_true, y_clst, matching)
        p = p.sum(axis=0) > 0  # Select MC samples regardless of true class
        targets = lcc_targets(
            dl, _y_true, y_clst, matching, n_true_classes=n_classes
        )
        for k, v in targets.items():
            targets[k] = v.to(model.device)
        lcc_data[sm] = LatentClusteringData(
            matching=matching,
            p=p,
            targets=targets,
            y_clst=y_clst,
        )

    return lcc_data


def full_dataloader_evaluation(
    model: BaseClassifier,
    dl: DataLoader,
    output_dir: str | Path,
    pca_dim: int | dict[str, int | None] | None = None,
    tqdm_style: Literal["notebook", "console", "none"] | None = None,
) -> None:
    """
    Evaluate model on whole dataset and saves latent representations and
    prediction batches in `output_dir`. The naming pattern is

        <output_dir> / <submodule>.<unique_id>.st

    and `unique_id` is a 32 character hexadecimal representation of a UUID4. The
    files are [Safetensors](https://huggingface.co/docs/safetensors/index)
    files. The content of a safetensor file is essentially a dictionary. In this
    case, the keys are
    * `""` (empty string): The latent representation tensor of this batch,
    * `"_idx"`: The indices of the samples in this batch.

    Args:
        model (BaseClassifier):
        dl (DataLoader):
        output_dir (str | Path):
        pca_dim (int | dict[str, int | None] | None, optional): If specified, a
            PCA is fitted and applied to the latent representations. This is
            useful when the latent dimensions are very large. However, it
            becomes necessary to iterate over the dataset twice, which more than
            doubles the execution time of this method. It is possible to specify
            a PCA dimension for each or some of the latent spaces.
        tqdm_style (Literal['notebook', 'console', 'none'] | None, optional):
            Defaults to `None`, mearning no progress bar.
    """
    if pca_dim is None:
        _fde_no_pca(
            model,
            dl,
            output_dir,
            tqdm_style=tqdm_style,
        )
    else:
        if isinstance(pca_dim, int):
            pca_dim = {sm: pca_dim for sm in model.lcc_submodules}
        _fde_pca(
            model,
            dl,
            output_dir,
            pca_dim=pca_dim,
            tqdm_style=tqdm_style,
        )


# TODO: Re-enable support for non-trivial class selection policies
def full_dataset_latent_clustering(
    model: BaseClassifier,
    output_dir: str | Path,
    # classes: list[int] | LCCClassSelection | None = None,
    tqdm_style: Literal["notebook", "console", "none"] | None = None,
) -> dict[str, LatentClusteringData]:
    """
    Distributed full train dataset latent clustering. Each rank evaluates part
    of the train dataset (accessing it through `model.trainer.train_dataloader`,
    which is different for every rank). Then, rank 0 loads all the latent
    representations and computes the latent clustering data (Louvain labels and
    matching), while all other ranks are waiting. Finally, the computed data is
    distributed across all ranks. Since holding all latent representation
    tensors in memory isn't realistic, some (aka. a shitload of) temporary
    tensor files are created in `<output_dir>/<split>`. See
    `full_dataset_evaluation`.

    Warning:
        The temporary tensor files created by this method are not deleted. You
        need to clean them up manually. Or don't if you want to keep them.

    Args:
        model (BaseClassifier):
        output_dir (str | Path):
        classes (list[int] | LCCClassSelection | None, optional): If specified,
            then only the specified true classes are considered for clustering
            (however, all samples are still evaluated regardless of class). Use
            this if there are too many true classes, or if the dataset is just
            too large to fit in memory (e.g. ImageNet). See also
            `nlnas.correction.LCC_CLASS_SELECTIONS`.
        tqdm_style (Literal['notebook', 'console', 'none'] | None, optional):

    Returns:
        A dictionary that maps a submodule name to its latent clustering data,
        see `nlnas.classifiers.LatentClusteringData`.
    """
    if not isinstance(model.trainer.train_dataloader, DataLoader):
        raise RuntimeError(
            "The model's trainer does not hold a valid training dataloader "
            "(pl.Train.train_dataloader)"
        )
    full_dataloader_evaluation(
        model,
        model.trainer.train_dataloader,
        output_dir=output_dir,
        tqdm_style=tqdm_style,
    )
    model.trainer.strategy.barrier()  # wait for every rank to finish eval.
    lcc_data: dict[str, LatentClusteringData] | None = None
    # Do actual clst. on rank 0 only; other ranks wait at the broadcast below
    if model.trainer.global_rank == 0:
        lcc_data = _fdlc_r0(model, output_dir, tqdm_style)
    if model.trainer.world_size >= 2:
        if lcc_data is not None:  # Only non-None on rank 0
            for d in lcc_data.values():
                for k, v in d.targets.items():
                    d.targets[k] = v.cpu()
        lcc_data = model.trainer.strategy.broadcast(lcc_data, src=0)
        assert isinstance(lcc_data, dict)
        for d in lcc_data.values():
            for k, v in d.targets.items():
                d.targets[k] = v.to(model.device)
    assert isinstance(lcc_data, dict)
    return lcc_data
