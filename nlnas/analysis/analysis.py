"""Main module"""

from pathlib import Path
from typing import Literal, Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import turbo_broccoli as tb
from loguru import logger as logging
from torch import Tensor, nn
from torchmetrics.functional.classification import multiclass_accuracy
from tqdm import tqdm

from ..classifiers import BaseClassifier
from ..correction import (
    class_otm_matching,
    louvain_communities,
    otm_matching_predicates,
)
from ..datasets import HuggingFaceDataset, dl_head, flatten_batches
from ..training import all_checkpoint_paths
from .plotting import louvain_clustering_plots, plot_latent_samples


def _acc(y_true: Tensor, y_pred: Tensor) -> float:
    """Helper function to compute micro-averaged multi-class accuracy"""
    return float(
        multiclass_accuracy(
            y_pred, y_true, num_classes=y_pred.shape[-1], average="micro"
        )
    )


def _ce(y_true: Tensor, y_pred: Tensor) -> float:
    """Helper function to compute cross-entropy loss"""
    return float(nn.functional.cross_entropy(y_pred, y_true.long()))


def analyse_ckpt(
    model: BaseClassifier | str | Path,
    submodule_names: list[str],
    dataset: HuggingFaceDataset,
    output_dir: str | Path,
    n_samples: int = 512,
    knn: int = 25,
    model_cls: Type[BaseClassifier] = BaseClassifier,
):
    """
    Analyses a model checkpoint. I can't be bothered to list everything this
    does. Suffice it to say that the main train-analyse workflow
    (`train_and_analyse all`) calls this method on every checkpoint obtained
    during training.

    Args:
        model (`nlnas.classifier.BaseClassifier` | str | Path): A model or a path
            to a model checkpoint
        submodule_names (list[str]): List or comma-separated list of
            submodule names. For example, the interesting submodules of
            `resnet18` are `maxpool`, layer1`, layer2`, layer3`, layer4` and
            `fc`
        dataset (HuggingFaceDataset):
        output_dir (str | Path):
        n_samples (int, optional):
        knn (int, optional): Number of neighbors to consider for computing the
            Louvain communities
        model_cls (Type[BaseClassifier], optional): The model class to use if
            `model` is a path to a checkpoint
    """
    output_dir = Path(output_dir)
    if not isinstance(model, BaseClassifier):
        model = model_cls.load_from_checkpoint(model)  # type: ignore
    assert isinstance(model, BaseClassifier)  # For typechecking
    model.eval()

    # EVALUATION
    h = tb.GuardedBlockHandler(output_dir / "eval" / "eval.json")
    for _ in h:
        logging.debug(
            "Evaluating model on test split (first {} samples)", n_samples
        )
        h.result = evaluate(model, submodule_names, dataset, n_samples)
    outputs: dict = h.result

    # UMAP EMBEDDINGS
    h = tb.GuardedBlockHandler(output_dir / "umap" / "umap.st")
    for _ in h:
        logging.debug("Computing UMAP embeddings")
        h.result = embed_latent_samples(outputs["z"])
    umap_embeddings: dict[str, np.ndarray] = h.result

    # UMAP EMBEDDINGS PLOTS
    h = tb.GuardedBlockHandler(
        output_dir / "umap" / "plots.json", load_if_skip=False
    )
    for _ in h:
        logging.debug("Plotting UMAP embeddings")
        h.result = plot_latent_samples(
            umap_embeddings, outputs["y_true"], h.file_path.parent
        )

    # LOUVAIN CLUSTERING FOR EACH LATENT SPACE
    progress = tqdm(
        outputs["z"].items(), desc="Louvain clustering", leave=False
    )
    for sm, z in progress:
        progress.set_postfix({"submodule": sm})

        # LOUVAIN CLUSTERING
        h = tb.GuardedBlockHandler(
            output_dir / "clustering" / sm / "cluster.json"
        )
        for _ in h:
            communities, y_louvain = louvain_communities(z, k=knn)
            matching = class_otm_matching(outputs["y_true"].numpy(), y_louvain)
            h.result = {
                "k": knn,
                "communities": communities,
                "y_louvain": y_louvain,
                "matching": matching,
            }
        y_louvain, matching = h.result["y_louvain"], h.result["matching"]

        # LOUVAIN CLUSTERING PLOTS
        h = tb.GuardedBlockHandler(
            output_dir / "clustering" / sm / "plots.json", load_if_skip=False
        )
        for _ in h:
            fig_scatter, fig_match = louvain_clustering_plots(
                z=umap_embeddings[sm],
                y_true=outputs["y_true"].numpy(),
                y_louvain=y_louvain,
                matching=matching,
                knn=knn,
                output_dir=h.file_path.parent,
            )
            h.result = {"scatter": fig_scatter, "match": fig_match}


def analyse_training(
    output_dir: str | Path,
    submodule_names: list[str],
    dataset: HuggingFaceDataset,
    n_samples: int = 512,
    knn: int = 25,
    model_cls: Type[BaseClassifier] = BaseClassifier,
):
    """
    Unlike `analyse_ckpt`, this method analyses the training as a whole. Again,
    I don't feel like explaining it all. Suffice it to say that the main
    trian-analyse workflow (`train_and_analyse all`) calls this method after
    calling `analyse_ckpt` on every checkpoint.

    Args:
        output_path (str | Path): e.g.
            `out.local/ft/cifar100/microsoft-resnet-18`
    """
    output_dir = (
        output_dir if isinstance(output_dir, Path) else Path(output_dir)
    )
    ckpts = all_checkpoint_paths(output_dir)
    logging.debug("Found {} checkpoints", len(ckpts))
    progress = tqdm(ckpts, desc="Analysing checkpoints", leave=False)
    for i, p in enumerate(progress):
        analyse_ckpt(
            model=p,
            submodule_names=submodule_names,
            dataset=dataset,
            output_dir=output_dir / str(i),
            n_samples=n_samples,
            knn=knn,
            model_cls=model_cls,
        )
    ckpt_an_dir = [output_dir / str(i) for i in range(len(ckpts))]

    # METRICS OF MISS/MATCH GROUPS
    h = tb.GuardedBlockHandler(output_dir / "metrics.csv")
    for _ in h:
        df = pd.DataFrame(
            columns=[
                "epoch",
                "layer",
                "ce_all",
                "ce_match",
                "ce_miss",
                "acc_all",
                "acc_match",
                "acc_miss",
            ]
        )
        progress = tqdm(ckpt_an_dir, desc="Collecting metrics", leave=False)
        for epoch, path in enumerate(progress):
            progress.set_postfix({"epoch": epoch})
            evaluations = tb.load_json(path / "eval" / "eval.json")
            y_true = evaluations["y_true"]
            y_pred = evaluations["y_pred"]
            ce_all, acc_all = _ce(y_true, y_pred), _acc(y_true, y_pred)
            for layer in evaluations["z"]:
                progress.set_postfix({"epoch": epoch, "layer": layer})
                clustering = tb.load_json(
                    path / "clustering" / layer / "cluster.json"
                )
                p1, p2, _, _ = otm_matching_predicates(
                    y_true, clustering["y_louvain"], clustering["matching"]
                )
                p_match = np.sum(p1 & p2, axis=0).astype(bool)
                df.loc[len(df)] = {
                    "epoch": epoch,
                    "layer": layer,
                    "ce_all": ce_all,
                    "ce_match": _ce(y_true[p_match], y_pred[p_match]),
                    "ce_miss": _ce(y_true[~p_match], y_pred[~p_match]),
                    "acc_all": acc_all,
                    "acc_match": _acc(y_true[p_match], y_pred[p_match]),
                    "acc_miss": _acc(y_true[~p_match], y_pred[~p_match]),
                }
        h.result = df

        # PLOTTING
        for prefix in ["ce", "acc"]:
            data = df[
                ["epoch", "layer"]
                + [f"{prefix}_all", f"{prefix}_match", f"{prefix}_miss"]
            ].copy()
            data.rename(
                columns={
                    "epoch": "epoch",
                    "layer": "layer",
                    f"{prefix}_all": "all",
                    f"{prefix}_match": "match",
                    f"{prefix}_miss": "miss",
                },
                inplace=True,
            )
            data = data.melt(
                id_vars=["epoch", "layer"],
                var_name="subset",
                value_name=prefix,
            )
            figure = sns.relplot(
                data=data,
                x="epoch",
                y=prefix,
                hue="subset",
                palette={"all": "black", "match": "blue", "miss": "red"},
                style="subset",
                dashes={"all": (1, 1), "match": "", "miss": ""},
                col="layer",
                kind="line",
            )
            figure.fig.savefig(output_dir / f"metrics_{prefix}.png")
            plt.clf()


def embed_latent_samples(
    z: dict[str, Tensor],
    device: Literal["cpu", "cuda"] | None = None,
) -> dict[str, np.ndarray]:
    """
    (Used as a step in `analyse_ckpt`) Embeds latent samples using UMAP. The
    latent embeddings are normalized too. This method isn't guarded.

    Args:
        z (dict[str, Tensor]): The dict of latent samples, a.k.a just a dict of
            tensors of shape `(N, ...)`
        device (Literal["cpu", "cuda"] | None, optional): If left to `None`,
            uses CUDA if it is available, otherwise falls back to CPU. Setting
            `cuda` while CUDA isn't available will silently fall back to CPU.

    Returns:
        A dict with the same keys and tensors of shape `(N, 2)`
    """
    if (device == "cuda" or device is None) and torch.cuda.is_available():
        from cuml import UMAP
    else:
        from umap import UMAP
    embeddings = {}
    progress = tqdm(z.items(), desc="UMAP embedding", leave=False)
    for k, v in progress:
        progress.set_postfix({"submodule": k})
        t = UMAP(n_components=2, metric="euclidean")
        e = t.fit_transform(v.flatten(1).numpy())
        e = (e - e.min(axis=0)) / (e.max(axis=0) - e.min(axis=0))
        embeddings[k] = e
    return embeddings


def evaluate(
    model: BaseClassifier,
    submodule_names: list[str],
    dataset: HuggingFaceDataset,
    n_samples: int = 512,
) -> dict:
    """
    (Used as a step in `analyse_ckpt`) Evaluates a model on the first
    `n_samples` samples of the *test* split of a given dataset. The return dict
    has the following structure:

        {
            "x": a tensor of shape (n_samples, C, H, W),
            "y_true": a label tensor of shape (n_samples,),
            "z": {
                submodule_name: a tensor of shape (n_samples, ...),
                ...
            }
            "y_pred": a logit tensor of shape (n_samples, n_classes)
        }

    This method isn't guarded.

    Args:
        model (BaseClassifier):
        submodule_names (list[str]):
        dataset (HuggingFaceDataset):
        n_samples (int, optional):
    """
    dataset.setup("test")
    batches = dl_head(dataset.test_dataloader(), n_samples)
    flat = flatten_batches(batches)
    out: dict[str, Tensor] = {}
    y_pred = model.forward_intermediate(
        batches, submodule_names, out, keep_gradients=False
    )
    return {
        "x": flat[model.image_key],
        "y_true": flat[model.label_key],
        "z": {k: torch.concat(v) for k, v in out.items()},  # type: ignore
        "y_pred": torch.concat(y_pred),  # type: ignore
    }
