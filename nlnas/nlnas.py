"""Main module"""

# TODO: Split module

import sys
import warnings
from glob import glob
from pathlib import Path
from typing import Type

import bokeh.layouts as bkl
import bokeh.plotting as bk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import regex as re
import seaborn as sns
import torch
import turbo_broccoli as tb
from bokeh.io import export_png
from loguru import logger as logging
from sklearn.metrics import accuracy_score, log_loss
from torch import Tensor
from tqdm import tqdm

from .classifiers import BaseClassifier
from .clustering import (
    class_otm_matching,
    louvain_communities,
    otm_matching_predicates,
)
from .datasets import TorchvisionDataset, dl_head
from .plotting import class_matching_plot, class_scatter, make_same_xy_range
from .training import all_checkpoint_paths, checkpoint_ves, train_model_guarded

if torch.cuda.is_available():
    from cuml import UMAP
else:
    from umap import UMAP


def _ce(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the cross entropy between two label vectors. Basically just calls
    [`sklearn.metrics.log_loss`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html)
    but hides warnings if the `y_pred` probability vector doesn't have rows
    that sum up to 1.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return log_loss(y_true, y_pred, labels=np.arange(y_pred.shape[-1]))


def _acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return accuracy_score(y_true, y_pred.argmax(axis=-1))


def _is_ckpt_analysis_dir(p: Path | str) -> bool:
    return Path(p).is_dir() and (re.match(r".*/\d+$", str(p)) is not None)


def analyse_ckpt(
    model: BaseClassifier | str | Path,
    submodule_names: list[str],
    dataset: pl.LightningDataModule | str,
    output_dir: str | Path,
    n_samples: int = 5000,
    model_cls: Type[BaseClassifier] | None = None,
):
    """
    Analyses a model checkpoint. I can't be bothered to list everything this
    does. Suffice it to say that the main trian-analyse workflow
    (`train_and_analyse all`) calls this method on every checkpoint obtained
    during training.

    Args:
        model (`nlnas.classifier.BaseClassifier` | str | Path): A model or a path
            to a model checkpoint
        submodule_names (list[str]): List or comma-separated list of
            submodule names. For example, the interesting submodules of
            `resnet18` are `maxpool`, layer1`, layer2`, layer3`, layer4` and
            `fc`
        dataset (pl.LightningDataModule | str): A lightning datamodule or the
            name of a torchvision dataset
        output_dir (str | Path):
        n_samples (int, optional):
    """
    output_dir = Path(output_dir)

    # LOAD MODEL IF NEEDED
    if not isinstance(model, BaseClassifier):
        model_cls = model_cls or BaseClassifier
        model = model_cls.load_from_checkpoint(model)
    assert isinstance(model, BaseClassifier)  # For typechecking
    model.eval()

    # EVALUATION
    h = tb.GuardedBlockHandler(output_dir / "eval" / "eval.json")
    for _ in h.guard():
        if not isinstance(dataset, pl.LightningDataModule):
            dataset = TorchvisionDataset(dataset)
        dataset.setup("fit")
        x_train, y_train = dl_head(dataset.train_dataloader(), n_samples)
        out: dict[str, Tensor] = {}
        y_pred = model.forward_intermediate(
            x_train,
            submodule_names,
            out,
        )
        h.result = {
            "x": x_train,
            "y_true": y_train,
            "z": out,
            "y_pred": y_pred,
        }
    outputs: dict = h.result
    x_train, y_train = outputs["x"], outputs["y_true"]

    # UMAP EMBEDDING
    h = tb.GuardedBlockHandler(output_dir / "umap" / "umap.st")
    for _ in h.guard():
        h.result = {}
        logging.debug("Computing UMAP embeddings")
        progress = tqdm(
            outputs["z"].items(), desc="UMAP embedding", leave=False
        )
        for sm, z in progress:
            progress.set_postfix({"submodule": sm})
            t = UMAP(n_components=2, metric="euclidean")
            e = t.fit_transform(z.flatten(1).numpy())
            e = (e - e.min(axis=0)) / (e.max(axis=0) - e.min(axis=0))
            h.result[sm] = e
    umap_embeddings: dict[str, np.ndarray] = h.result

    # PLOTTING
    h = tb.GuardedBlockHandler(output_dir / "umap" / "plots.json")
    for _ in h.guard():
        h.result = {}
        logging.debug("Plotting UMAP embeddings")
        progress = tqdm(
            umap_embeddings.items(), desc="UMAP plotting", leave=False
        )
        for sm, e in progress:
            progress.set_postfix({"submodule": sm})
            figure = bk.figure(title=sm, toolbar_location=None)
            class_scatter(figure, e, y_train.numpy())
            h.result[sm] = figure
            export_png(figure, filename=h.file_path.parent / (sm + ".png"))

    # LOUVAIN CLUSTERING
    progress = tqdm(
        outputs["z"].items(), desc="Louvain clustering", leave=False
    )
    knn = 25
    for sm, z in progress:
        progress.set_postfix({"submodule": sm})

        # CLUSTERING
        h = tb.GuardedBlockHandler(
            output_dir / "clustering" / sm / "cluster.json"
        )
        for _ in h.guard():
            communities, y_louvain = louvain_communities(z, k=knn)
            matching = class_otm_matching(outputs["y_true"].numpy(), y_louvain)
            h.result = {
                "k": knn,
                "communities": communities,
                "y_louvain": y_louvain,
                "matching": matching,
            }
        y_louvain, matching = h.result["y_louvain"], h.result["matching"]

        h = tb.GuardedBlockHandler(
            output_dir / "clustering" / sm / "plots.json"
        )
        for _ in h.guard():
            h.result = {}

            # Side-by-side class scatter
            fig_true = bk.figure(title="Ground truth")
            class_scatter(
                fig_true,
                umap_embeddings[sm],
                outputs["y_true"].numpy(),
                palette="viridis",
            )

            fig_louvain = bk.figure(
                title=(
                    f"Louvain communities ({y_louvain.max() + 1}), k = {knn}"
                ),
            )
            class_scatter(fig_louvain, umap_embeddings[sm], y_louvain)
            make_same_xy_range(fig_true, fig_louvain)
            h.result["scatter"] = bkl.row(fig_true, fig_louvain)

            # Class matching plot
            h.result["match"] = class_matching_plot(
                umap_embeddings[sm],
                outputs["y_true"].numpy(),
                y_louvain,
                matching,
            )

            # EXPORT
            for k, v in h.result.items():
                export_png(v, filename=h.file_path.parent / (k + ".png"))


def analyse_training(output_dir: str | Path, last_epoch: int | None = None):
    """
    Unlike `analyse_ckpt`, this method analyses the training as a whole. Again,
    I don't feel like explaining it all. Suffice it to say that the main
    trian-analyse workflow (`train_and_analyse all`) calls this method after
    calling `analyse_ckpt` on every checkpoint.

    Args:
        output_path (str | Path): e.g. `./out/resnet18/cifar10/version_1/`
        last_epoch (int, optional): Only considers training epoch up to that
            epoch
    """
    output_dir = (
        output_dir if isinstance(output_dir, Path) else Path(output_dir)
    )
    ckpt_analysis_dirs = list(
        map(Path, filter(_is_ckpt_analysis_dir, glob(str(output_dir / "*"))))
    )
    last_epoch = last_epoch or len(ckpt_analysis_dirs) - 1

    # UMAP PLOT
    # umap_all_path = output_dir / "umap_all.png"
    # if not umap_all_path.exists():
    #     rows, epochs = [], np.linspace(0, last_epoch, num=10, dtype=int)
    #     logging.info("Consolidating UMAP plots")
    #     progress = tqdm(
    #         ckpt_analysis_dirs, desc="UMAP summary plot", leave=False
    #     )
    #     for epoch, path in enumerate(progress):
    #         if not epoch in epochs:
    #             continue
    #         progress.set_postfix({"epoch": epoch})
    #         document = tb.load_json(path / "umap" / "plots.json")
    #         for figure in document.values():
    #             figure.height, figure.width = 200, 200
    #             figure.grid.visible, figure.axis.visible = False, False
    #         rows.append(list(document.values()))
    #     figure = bk.gridplot(rows)
    #     logging.debug("Sleeping for 5s before rendering to file")
    #     sleep(5)
    #     export_png(figure, filename=umap_all_path)

    # METRICS OF MISS/MATCH GROUPS
    h = tb.GuardedBlockHandler(output_dir / "metrics.csv")
    for _ in h.guard():
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
        progress = tqdm(
            ckpt_analysis_dirs, desc="Collecting metrics", leave=False
        )
        for epoch, path in enumerate(progress):
            progress.set_postfix({"epoch": epoch})
            evaluations = tb.load_json(path / "eval" / "eval.json")
            y_true = evaluations["y_true"].numpy()
            y_pred = evaluations["y_pred"].numpy()
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


def train_and_analyse_all(
    model: BaseClassifier,
    submodule_names: list[str],
    dataset: pl.LightningDataModule | str,
    output_dir: str | Path,
    model_name: str | None = None,
    n_samples: int = 5000,
    strategy: str = "auto",
):
    """
    Trains a model and performs a separability analysis on ALL model
    checkpoints (1 per training epoch).

    1. Train `model` on `dataset`;
    2. Call `analyse_ckpt` on every checkpoint;
    3. Call `analyse_training`.

    Args:
        model (`nlnas.classifier.BaseClassifier`):
        submodule_names (list[str]): List or comma-separated list of
            submodule names. For example, the interesting submodules of
            `resnet18` are `maxpool, layer1, layer2, layer3, layer4, fc`
        dataset (pl.LightningDataModule | str):
        output_dir (str | Path):
        model_kwargs (dict[str, Any], optional): Defaults to None.
        model_name (str, optional): Model name override (for naming
            directories and for logging). If left to `None`, is set to the
            lower case class name, i.e. `model.__class__.__name__.lower()`.
        n_samples (int, optional): The analysis part of this workflow requires
            evaluating the model on some samples from the dataset. Defaults to
            5000 samples.
        strategy (str, optional): Training strategy to use. See
            https://lightning.ai/docs/pytorch/stable/extensions/strategy.html#selecting-a-built-in-strategy
    """
    model_name = model_name or model.__class__.__name__.lower()
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    if isinstance(dataset, str):
        dataset = TorchvisionDataset(dataset)
    _, best_ckpt = train_model_guarded(
        model,
        dataset,
        output_dir / "model",
        name=model_name,
        max_epochs=512,
        strategy=strategy,
        # early_stopping_kwargs={
        #     "monitor": "val/loss",
        #     "patience": 20,
        #     "mode": "min",
        # },
    )
    if model.global_rank != 0:
        sys.exit(0)
    version, best_epoch, _ = checkpoint_ves(best_ckpt)
    p = (
        output_dir
        / "model"
        / "tb_logs"
        / model_name
        / f"version_{version}"
        / "checkpoints"
    )
    logging.info("{}: Analyzing epochs", model_name)
    progress = tqdm(all_checkpoint_paths(p), leave=False)
    for i, ckpt in enumerate(progress):
        analyse_ckpt(
            model=ckpt,
            model_cls=type(model),
            submodule_names=submodule_names,
            dataset=dataset,
            output_dir=output_dir / f"version_{version}" / str(i),
            n_samples=n_samples,
        )
    logging.info("{}: Analyzing training", model_name)
    analyse_training(
        output_dir / f"version_{version}",
        last_epoch=best_epoch,
    )
