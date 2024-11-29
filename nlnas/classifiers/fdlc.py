"""
This module revolves around the implementation of
`full_dataset_latent_clustering`.
"""

from pathlib import Path
from uuid import uuid4

import torch
import torch.distributed
from safetensors import torch as st
from torch import Tensor
from torch.utils.data import DataLoader

from ..correction import ExactLCCLoss, class_otm_matching, louvain_communities
from ..datasets import BatchedTensorDataset
from ..utils import TqdmStyle, make_tqdm
from .base import BaseClassifier, LatentClusteringData


def _fde_pca(
    model: BaseClassifier,
    dl: DataLoader,
    output_dir: str | Path,
    pca_dim: dict[str, int | None],
    tqdm_style: TqdmStyle = None,
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

    ws, gr = model.trainer.world_size, model.trainer.global_rank
    output_dir, tqdm = Path(output_dir), make_tqdm(tqdm_style)
    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        model.eval()
        for batch in tqdm(dl, f"[Rank {gr}/{ws}] Fitting PCA(s)"):
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
        for batch in tqdm(dl, f"[Rank {gr}/{ws}] Evaluating"):
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
    tqdm_style: TqdmStyle = None,
) -> None:
    """
    `full_dataset_evaluation` in the case where no PCA dim-redux is to be
    applied. See `full_dataset_evaluation` for the precise meaning of the
    arguments.
    """
    ws, gr = model.trainer.world_size, model.trainer.global_rank
    output_dir, tqdm = Path(output_dir), make_tqdm(tqdm_style)
    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        model.eval()
        for batch in tqdm(dl, f"[Rank {gr}/{ws}] Evaluating"):
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


def _construct_latent_data(
    model: BaseClassifier, output_dir: str | Path, tqdm_style: TqdmStyle = None
) -> dict[str, LatentClusteringData]:
    """
    Full dataset latent clustering, to only be executed on rank 0. It is assumed
    that the whole dataset has already been evaluated.
    """
    lcc_kwargs = model.hparams.get("lcc_kwargs", {})
    lcc_data: dict[str, LatentClusteringData] = {}
    tqdm_style = tqdm_style if model.trainer.global_rank == 0 else None

    # Step 0: Get the true label vector in the same order as in the original
    # dataset (so ordered by absolute index). BatchedTensorDataset.load is
    # potentially i/o expensive so it's only done on rank 0
    if model.trainer.global_rank == 0:
        y_true, idx = BatchedTensorDataset(output_dir, prefix="y_true").load()
        # ↓ make sure y_true is in the same order as in the batches
        y_true = y_true[idx.argsort()]
    else:
        y_true = torch.empty(0)
    y_true = model.trainer.strategy.broadcast(y_true, src=0)

    progress = make_tqdm(tqdm_style)(model.lcc_submodules, "Clustering")
    for sm in progress:
        progress.set_postfix(submodule=sm)
        ds = BatchedTensorDataset(output_dir, prefix=sm)

        # Step 1: Get a true label vector with the same order as in the batches
        # of ds. extract_idx is potentially i/o expensive so it's only done on
        # rank 0
        if model.trainer.global_rank == 0:
            _, idx = ds.extract_idx(tqdm_style)
            _y_true = y_true[idx]  # Make sure the order matches that of the ds
        else:
            _y_true = torch.empty(0)
        _y_true = model.trainer.strategy.broadcast(_y_true, src=0)

        # Step 2: Distributed louvain community detection
        _, y_clst = louvain_communities(
            ds,
            k=lcc_kwargs.get("k", 5),
            strategy=model.trainer.strategy,
            device=model.device,
            tqdm_style=tqdm_style,
        )

        # Step 3: Computing matching. class_otm_matching might not be
        # deterministic, so it's computed on rank 0 only
        if model.trainer.global_rank == 0:
            matching = class_otm_matching(_y_true, y_clst)
        else:
            matching = {}
        matching = model.trainer.strategy.broadcast(matching, src=0)

        # Step 4: Create the loss object
        loss = ExactLCCLoss(
            n_classes=model.hparams["n_classes"],
            k=lcc_kwargs.get("k", 5),
            tqdm_style=tqdm_style,
            strategy=model.trainer.strategy,
        )
        dl = DataLoader(ds, batch_size=256, num_workers=4)
        loss.update(dl, _y_true, y_clst, matching)
        loss.on_before_sync()
        loss.sync()
        loss.on_after_sync()
        lcc_data[sm] = LatentClusteringData(loss=loss, y_clst=y_clst)
    return lcc_data


def full_dataloader_evaluation(
    model: BaseClassifier,
    dl: DataLoader,
    output_dir: str | Path,
    pca_dim: int | dict[str, int | None] | None = None,
    tqdm_style: TqdmStyle = None,
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
        tqdm_style (TqdmStyle, optional):
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
    model: BaseClassifier, output_dir: str | Path, tqdm_style: TqdmStyle = None
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
        tqdm_style (TqdmStyle, optional):

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
    model.trainer.strategy.barrier()  # Wait for every rank to finish eval.
    lcc_data = _construct_latent_data(model, output_dir, tqdm_style)
    return lcc_data
