"""Main module"""

from glob import glob
from pathlib import Path
from time import sleep
from typing import Type

import bokeh.plotting as bk
import bokeh.layouts as bkl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import regex as re
import seaborn as sns
import turbo_broccoli as tb
from bokeh.io import export_png
from loguru import logger as logging
from torch import Tensor
from tqdm import tqdm
from umap import UMAP

from .classifier import Classifier
from .clustering import (
    louvain_communities,
    class_otm_matching,
)
from .plotting import class_scatter, class_matching_plot, make_same_xy_range
from .separability import gdv, label_variation, mean_ggd
from .training import all_checkpoint_paths, checkpoint_ves, train_model_guarded
from .tv_dataset import TorchvisionDataset
from .utils import get_first_n


def _is_ckpt_analysis_dir(p: Path | str) -> bool:
    return Path(p).is_dir() and (re.match(r".*/\d+$", str(p)) is not None)


def analyse_ckpt(
    model: Classifier | str | Path,
    submodule_names: list[str],
    dataset: pl.LightningDataModule | str,
    output_dir: str | Path,
    n_samples: int = 5000,
    model_cls: Type[Classifier] | None = None,
):
    """
    Full separability analysis and plottings

    Args:
        model (Classifier | str | Path): A model or a path to a
            model checkpoint
        submodule_names (list[str]): List or comma-separated list of
            submodule names. For example, the interesting submodules of
            `resnet18` are `maxpool, layer1, layer2, layer3, layer4, fc`
        dataset (pl.LightningDataModule | str): A lightning datamodule or the
            name of a torchvision dataset
        output_dir (str | Path):
        n_samples (int, optional):
    """
    output_dir = Path(output_dir)

    # LOAD MODEL IF NEEDED
    if not isinstance(model, Classifier):
        model_cls = model_cls or Classifier
        model = model_cls.load_from_checkpoint(model)
    assert isinstance(model, Classifier)  # For typechecking
    model.eval()

    # EVALUATION
    h = tb.GuardedBlockHandler(output_dir / "eval" / "eval.json")
    for _ in h.guard():
        if not isinstance(dataset, pl.LightningDataModule):
            dataset = TorchvisionDataset(dataset)
        dataset.setup("fit")
        x_train, y_train = get_first_n(dataset.train_dataloader(), n_samples)
        out: dict[str, Tensor] = {}
        model.forward_intermediate(
            x_train,
            submodule_names,
            out,
        )
        h.result = {"x": x_train, "y": y_train, "z": out}
    outputs: dict = h.result
    x_train, y_train = outputs["x"], outputs["y"]

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
            e = t.fit_transform(z.flatten(1))
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
            export_png(figure, filename=h.output_path.parent / (sm + ".png"))

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
            (
                communities,
                y_louvain,
                kd_tree,
                knn_dist,
                knn_idx,
            ) = louvain_communities(
                z.flatten(1).numpy(), k=knn, scaling="standard"
            )
            matching = class_otm_matching(outputs["y"].numpy(), y_louvain)
            h.result = {
                "k": knn,
                "communities": communities,
                "y_louvain": y_louvain,
                "kd_tree": kd_tree,
                "knn_dist": knn_dist,
                "knn_idx": knn_idx,
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
                outputs["y"].numpy(),
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
                umap_embeddings[sm], outputs["y"].numpy(), y_louvain, matching
            )

            # EXPORT
            for k, v in h.result.items():
                export_png(v, filename=h.output_path.parent / (k + ".png"))


def analyse_training(
    output_dir: str | Path,
    lv_k: int = 10,
    last_epoch: int | None = None,
):
    """
    For now only plot LV scores per epoch and per submodule

    Args:
        output_path (str | Path): e.g. `./out/resnet18/cifar10/version_1/`
        n_samples (int): Sorry it's not inferred ¯\\_(ツ)_/¯
        lv_k (int, optional): $k$ hyperparameter to compute LV
        last_epoch (int, optional): If specified, only plot LV curves up to
            that epoch
    """
    output_dir = (
        output_dir if isinstance(output_dir, Path) else Path(output_dir)
    )
    ckpt_analysis_dirs = list(
        filter(_is_ckpt_analysis_dir, glob(str(output_dir / "*")))
    )
    last_epoch = last_epoch or len(ckpt_analysis_dirs) - 1

    # LV COMPUTATION
    h = tb.GuardedBlockHandler(output_dir / "lv.csv")
    for _ in h.guard():
        data = []
        logging.info("Computing LVs")
        progress = tqdm(ckpt_analysis_dirs, desc="LV", leave=False)
        for epoch, path in enumerate(progress):
            evaluations = tb.load_json(Path(path) / "eval" / "eval.json")
            for sm, z in evaluations["z"].items():
                progress.set_postfix({"epoch": epoch, "submodule": sm})
                v = float(label_variation(z, evaluations["y"], k=lv_k))
                data.append([epoch, sm, v])
        df = pd.DataFrame(data, columns=["epoch", "submodule", "lv"])
        h.result = df

        # PLOTTING
        e = np.linspace(0, last_epoch, num=5, dtype=int)
        figure = sns.lineplot(
            df[df["epoch"].isin(e)],
            x="submodule",
            y="lv",
            hue="epoch",
            size="epoch",
        )
        figure.set(title="Label variation by epoch")
        figure.set_xticklabels(
            figure.get_xticklabels(),
            rotation=45,
            rotation_mode="anchor",
            ha="right",
        )
        figure.get_figure().savefig(output_dir / "lv_epoch.png")
        plt.clf()

    # GDV COMPUTATION
    h = tb.GuardedBlockHandler(output_dir / "gdv.csv")
    for _ in h.guard():
        data = []
        logging.info("Computing GDVs")
        progress = tqdm(ckpt_analysis_dirs, desc="GDV", leave=False)
        for epoch, path in enumerate(progress):
            evaluations = tb.load_json(Path(path) / "eval" / "eval.json")
            for sm, z in evaluations["z"].items():
                progress.set_postfix({"epoch": epoch, "submodule": sm})
                v = float(gdv(z, evaluations["y"]))
                data.append([epoch, sm, v])
        df = pd.DataFrame(data, columns=["epoch", "submodule", "gdv"])
        h.result = df

        # PLOTTING
        e = np.linspace(0, last_epoch, num=5, dtype=int)
        figure = sns.lineplot(
            df[df["epoch"].isin(e)],
            x="submodule",
            y="gdv",
            hue="epoch",
            size="epoch",
        )
        figure.set(title="GDV by epoch")
        figure.set_xticklabels(
            figure.get_xticklabels(),
            rotation=45,
            rotation_mode="anchor",
            ha="right",
        )
        figure.get_figure().savefig(output_dir / "gdv_epoch.png")
        plt.clf()

    # GRASSMANNIAN DISTANCE COMPUTATION
    h = tb.GuardedBlockHandler(output_dir / "gr.csv")
    for _ in h.guard():
        data = []
        logging.info("Computing Grassmannian geodesic distances")
        progress = tqdm(ckpt_analysis_dirs, desc="GGD", leave=False)
        for epoch, path in enumerate(progress):
            evaluations = tb.load_json(Path(path) / "eval" / "eval.json")
            for sm, z in evaluations["z"].items():
                progress.set_postfix({"epoch": epoch, "submodule": sm})
                v = float(mean_ggd(z.flatten(1), evaluations["y"]))
                data.append([epoch, sm, v])
        df = pd.DataFrame(data, columns=["epoch", "submodule", "gr"])
        h.result = df

        # PLOTTING
        # evaluations = tb.load_json(
        #     Path(ckpt_analysis_dirs[0]) / "eval" / "eval.json"
        # )
        progress = tqdm(
            evaluations["z"].keys(), desc="GGD plotting", leave=False
        )
        for sm in progress:
            progress.set_postfix({"submodule": sm})
            figure = sns.lineplot(df[df["submodule"] == sm], x="epoch", y="gr")
            figure.axvline(last_epoch, linestyle=":", color="gray")
            figure.set(title=f"Mean Grass. dst. for {sm}")
            figure.get_figure().savefig(output_dir / f"gr_{sm}.png")
            plt.clf()

    umap_all_path = output_dir / "umap_all.png"
    if not umap_all_path.exists():
        rows, epochs = [], np.linspace(0, last_epoch, num=10, dtype=int)
        logging.info("Consolidating UMAP plots")
        progress = tqdm(
            ckpt_analysis_dirs, desc="UMAP summary plot", leave=False
        )
        for epoch, path in enumerate(progress):
            if not epoch in epochs:
                continue
            progress.set_postfix({"epoch": epoch})
            document = tb.load_json(Path(path) / "umap" / "plots.json")
            for figure in document.values():
                figure.height, figure.width = 200, 200
                figure.grid.visible, figure.axis.visible = False, False
            rows.append(list(document.values()))
        figure = bk.gridplot(rows)
        logging.debug("Sleeping for 5s before rendering to file")
        sleep(5)
        export_png(figure, filename=umap_all_path)


def train_and_analyse_all(
    model: Classifier,
    submodule_names: list[str],
    dataset: pl.LightningDataModule | str,
    output_dir: str | Path,
    model_name: str | None = None,
    n_samples: int = 5000,
    strategy: str = "ddp",
):
    """
    Trains a model and performs a separability analysis (see
    `nlnas.nlnas.analyse`) on ALL models, obtained at the end of each epochs.

    Args:
        model (Classifier):
        submodule_names (list[str]): List or comma-separated list of
            submodule names. For example, the interesting submodules of
            `resnet18` are `maxpool, layer1, layer2, layer3, layer4, fc`
        dataset (pl.LightningDataModule | str):
        output_dir (str | Path):
        model_kwargs (dict[str, Any], optional): Defaults to None.
        model_name (str, optional): Model name override (for naming
            directories and for logging). If left to `None`, is set to the
            lower case class name, i.e. `model.__class__.__name__.lower()`.
        n_samples (int, optional): Defaults to 5000.
        strategy (str, optional): Training strategy to use.
    """
    # tb.set_max_nbytes(1000)  # Ensure artefacts
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
    )
    if model.global_rank != 0:
        return
    version, best_epoch, _ = checkpoint_ves(best_ckpt)
    p = (
        output_dir
        / "model"
        / "tb_logs"
        / model_name
        / f"version_{version}"
        / "checkpoints"
    )
    logging.info("Analyzing epochs")
    progress = tqdm(all_checkpoint_paths(p), leave=False)
    # progress = tqdm([], leave=False)
    for i, ckpt in enumerate(progress):
        analyse_ckpt(
            model=ckpt,
            model_cls=type(model),
            submodule_names=submodule_names,
            dataset=dataset,
            output_dir=output_dir / f"version_{version}" / str(i),
            n_samples=n_samples,
        )
    logging.info("Analyzing training")
    analyse_training(
        output_dir / f"version_{version}",
        last_epoch=best_epoch,
    )
