"""Main module"""

import random
from glob import glob
from itertools import combinations
from pathlib import Path
from time import sleep
from typing import Type

import bokeh.plotting as bk
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
from .plotting import class_scatter
from .separability import gdv, label_variation, mean_ggd, pairwise_svc_scores
from .training import all_checkpoint_paths, checkpoint_ves, train_model_guarded
from .tv_dataset import TorchvisionDataset
from .utils import get_first_n

MAX_CLASS_PAIRS = 200
"""
In `analyse_ckpt`, if `svc_separability` is `True`, a SVC-based
separability scoring is computed. Specifically, for each pair of classes, we
compute how well separated they are by fitting a SVC and computing the score.
If there are many classes, the number of pairs can be very large. If this
number is greater than `MAX_CLASS_PAIRS`, then only `MAX_CLASS_PAIRS` pairs are
chosen at random (among all possible pairs) for the separability scoring.
"""


def _is_ckpt_analysis_dir(p: Path | str) -> bool:
    return Path(p).is_dir() and (re.match(r".*/\d+$", str(p)) is not None)


def analyse_ckpt(
    model: Classifier | str | Path,
    submodule_names: list[str],
    dataset: pl.LightningDataModule | str,
    output_dir: str | Path,
    n_samples: int = 5000,
    model_cls: Type[Classifier] | None = None,
    umap: bool = True,
    svc_separability: bool = True,
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
        umap (bool, optional): Wether to compute UMAP embeddings and plots
        svc_separability (bool, optional): Wether to compute the UMAP-SVC
            separability scores and plots. If set to `True`, overrides `umap`.
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

    # UMAP
    if umap or svc_separability:
        # EMBEDDING
        h = tb.GuardedBlockHandler(output_dir / "umap" / "umap.json")
        for _ in h.guard():
            h.result = {}
            logging.debug("Computing UMAP embeddings")
            progress = tqdm(outputs["z"].items(), leave=False)
            for k, m in progress:
                progress.set_postfix({"submodule": k})
                t = UMAP(n_components=2, metric="euclidean")
                e = t.fit_transform(m.flatten(1))
                e = (e - e.min(axis=0)) / (e.max(axis=0) - e.min(axis=0))
                h.result[k] = e
        umap_embeddings: dict[str, np.ndarray] = h.result

        # PLOTTING
        h = tb.GuardedBlockHandler(output_dir / "umap" / "plots.json")
        for _ in h.guard():
            h.result = {}
            logging.debug("Plotting UMAP embeddings")
            progress = tqdm(umap_embeddings.items(), leave=False)
            for k, e in progress:
                progress.set_postfix({"submodule": k})
                figure = bk.figure(title=k, toolbar_location=None)
                class_scatter(figure, e, y_train.numpy(), "viridis")
                h.result[k] = figure
                export_png(
                    figure, filename=h.output_path.parent / (k + ".png")
                )

    # SEPARABILITY SCORE AND PLOTTING
    if svc_separability:
        n_classes = len(np.unique(y_train))
        class_idx_pairs = list(combinations(range(n_classes), 2))
        if (n_classes * (n_classes - 1) / 2) > MAX_CLASS_PAIRS:
            class_idx_pairs = random.sample(class_idx_pairs, MAX_CLASS_PAIRS)
        h = tb.GuardedBlockHandler(output_dir / "svc" / "pairwise_rbf.json")
        # h = tb.GuardedBlockHandler(output_dir / "svc" / "pairwise_linear.json")
        # h = tb.GuardedBlockHandler(output_dir / "svc" / "full_linear.json")
        for _ in h.guard():
            h.result = {}
            # PAIRWISE RBF
            logging.debug("Fitting SVCs")
            progress = tqdm(umap_embeddings.items(), leave=False)
            for k, e in progress:
                progress.set_postfix({"submodule": k})
                h.result[k] = pairwise_svc_scores(
                    e, y_train, MAX_CLASS_PAIRS, kernel="rbf"
                )
                h.result[k] = pairwise_svc_scores(
                    e, y_train, MAX_CLASS_PAIRS, kernel="linear"
                )

            # FULL LINEAR
            # for k, e in outputs.items():
            #     logging.debug("Fitting SVC for outputs of submodule '{}'", k)
            #     a = e.flatten(1).numpy()
            #     svc = SVC(kernel="linear").fit(a, y)
            #     h.result[k] = {"svc": svc, "score": svc.score(a, y)}

            # Plotting is done here to be in the guarded block
            logging.debug("Plotting separability scores")
            scores = [
                [k, np.mean([d["score"] for d in v])]
                for k, v in h.result.items()
            ]
            # scores = [[k, v["score"]] for k, v in h.result.items()]
            df = pd.DataFrame(scores, columns=["submodule", "score"])
            figure = sns.lineplot(df, x="submodule", y="score")
            figure.set(title="Linear separability score")
            figure.set_xticklabels(
                figure.get_xticklabels(),
                rotation=45,
                rotation_mode="anchor",
                ha="right",
            )
            figure.get_figure().savefig(
                h.output_path.parent / "separability.png"
            )
            plt.clf()


def analyse_training(
    output_dir: str | Path,
    lv_k: int = 10,
    last_epoch: int | None = None,
    umap: bool = False,
    # svc_separability: bool = True,
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
    lv_csv_path = output_dir / "lv.csv"
    if not lv_csv_path.exists():
        data = []
        logging.info("Computing LVs")
        progress = tqdm(ckpt_analysis_dirs, leave=False)
        for epoch, path in enumerate(progress):
            evaluations = tb.load_json(Path(path) / "eval" / "eval.json")
            for sm, z in evaluations["z"].items():
                progress.set_postfix({"epoch": epoch, "submodule": sm})
                v = float(label_variation(z, evaluations["y"], k=lv_k))
                data.append([epoch, sm, v])
        lvs = pd.DataFrame(data, columns=["epoch", "submodule", "lv"])
        lvs.to_csv(lv_csv_path)
        # PLOTTING
        e = np.linspace(0, last_epoch, num=5, dtype=int)
        figure = sns.lineplot(
            lvs[lvs["epoch"].isin(e)],
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
    gdv_csv_path = output_dir / "gdv.csv"
    if not gdv_csv_path.exists():
        data = []
        logging.info("Computing GDVs")
        progress = tqdm(ckpt_analysis_dirs, leave=False)
        for epoch, path in enumerate(progress):
            evaluations = tb.load_json(Path(path) / "eval" / "eval.json")
            for sm, z in evaluations["z"].items():
                progress.set_postfix({"epoch": epoch, "submodule": sm})
                v = float(gdv(z, evaluations["y"]))
                data.append([epoch, sm, v])
        gdvs = pd.DataFrame(data, columns=["epoch", "submodule", "gdv"])
        gdvs.to_csv(gdv_csv_path)
        # PLOTTING
        e = np.linspace(0, last_epoch, num=5, dtype=int)
        figure = sns.lineplot(
            gdvs[gdvs["epoch"].isin(e)],
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
    gr_csv_path = output_dir / "gr.csv"
    if not gr_csv_path.exists():
        data = []
        logging.info("Computing Grassmannian geodesic distances")
        progress = tqdm(ckpt_analysis_dirs, leave=False)
        for epoch, path in enumerate(progress):
            evaluations = tb.load_json(Path(path) / "eval" / "eval.json")
            for sm, z in evaluations["z"].items():
                progress.set_postfix({"epoch": epoch, "submodule": sm})
                v = float(mean_ggd(z.flatten(1), evaluations["y"]))
                data.append([epoch, sm, v])
        grs = pd.DataFrame(data, columns=["epoch", "submodule", "gr"])
        grs.to_csv(gr_csv_path)
        # PLOTTING
        # evaluations = tb.load_json(
        #     Path(ckpt_analysis_dirs[0]) / "eval" / "eval.json"
        # )
        progress = tqdm(evaluations["z"].keys(), desc="Plotting", leave=False)
        for sm in progress:
            progress.set_postfix({"submodule": sm})
            figure = sns.lineplot(
                grs[grs["submodule"] == sm], x="epoch", y="gr"
            )
            figure.axvline(last_epoch, linestyle=":", color="gray")
            figure.set(title=f"Mean Grass. dst. for {sm}")
            figure.get_figure().savefig(output_dir / f"gr_{sm}.png")
            plt.clf()

    umap_all_path = output_dir / "umap_all.png"
    if umap and not umap_all_path.exists():
        rows, epochs = [], np.linspace(0, last_epoch, num=10, dtype=int)
        logging.info("Consolidating UMAP plots")
        progress = tqdm(ckpt_analysis_dirs, leave=False)
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
    umap: bool = True,
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
        umap (bool, optional): Wether to compute and plot UMAP embeddings.
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
            umap=umap,
            svc_separability=False,
        )
    logging.info("Analyzing training")
    analyse_training(
        output_dir / f"version_{version}",
        last_epoch=best_epoch,
        umap=umap,
    )
