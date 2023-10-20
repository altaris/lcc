"""Main module"""

import random
import shutil
from glob import glob
from pathlib import Path
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
from phate import PHATE
from sklearn.manifold import TSNE
from torch import Tensor
from tqdm import tqdm

from .classifier import Classifier
from .pdist import pdist
from .plotting import class_scatter
from .separability import label_variation, pairwise_svc_scores
from .training import all_ckpt_paths, checkpoint_ves, train_model_guarded
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
    max_class_pairs: int = 200,
    model_cls: Type[Classifier] | None = None,
    tsne: bool = True,
    tsne_svc_separability: bool = True,
    phate: bool = False,
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
        max_class_pairs (int, optional):
        tsne (bool, optional): Wether to compute TSNE embeddings and plots
        tsne_svc_separability (bool, optional): Wether to compute the TSNE-SVC
            separability scores and plots. If set to `True`, overrides `tsne`.
        phate (bool, optional): Wether to compute the PHATE embeddings and plots
    """
    output_dir = Path(output_dir)

    # LOAD MODEL IF NEEDED
    if not isinstance(model, Classifier):
        logging.info("Analysing checkpoint {}", str(model))
        model_cls = model_cls or Classifier
        model = model_cls.load_from_checkpoint(model)
    assert isinstance(model, Classifier)  # For typechecking
    model.eval()

    # LOAD DATASET IF NEEDED
    if not isinstance(dataset, pl.LightningDataModule):
        dataset = TorchvisionDataset(dataset)
    dataset.setup("fit")
    x_train, y_train = get_first_n(dataset.train_dataloader(), n_samples)

    # EVALUATION
    h = tb.GuardedBlockHandler(output_dir / "eval" / "eval.json")
    for _ in h.guard():
        logging.debug("Evaluating model")
        out: dict[str, Tensor] = {}
        model.eval()
        model.forward_intermediate(
            x_train,
            submodule_names,
            out,
        )
        h.result = {k: v.contiguous() for k, v in out.items()}  # hotfix
    outputs: dict[str, Tensor] = h.result

    # TSNE
    if tsne or tsne_svc_separability:
        # COMPUTE DISTANCE MATRICES
        h = tb.GuardedBlockHandler(output_dir / "pdist" / "pdist.json")
        chunk_path = h.output_path.parent / "chunks"
        for _ in h.guard():
            h.result = {}
            for k, z in outputs.items():
                logging.debug(
                    "Computing distance matrix for outputs of submodule '{}'",
                    k,
                )
                h.result[k] = pdist(
                    z.flatten(1).numpy(),
                    chunk_size=int(n_samples / 10),
                    chunk_path=chunk_path / k,
                )
        if chunk_path.is_dir():
            logging.debug("Removing pdist chuncks directory '{}'", chunk_path)
            try:
                shutil.rmtree(chunk_path)
            except OSError as err:
                logging.error(
                    "Could not remove chunks directory '{}': {}",
                    chunk_path,
                    err,
                )
        distance_matrices: dict[str, np.ndarray] = h.result

        # EMBEDDING
        h = tb.GuardedBlockHandler(output_dir / "tsne" / "tsne.json")
        for _ in h.guard():
            h.result = {}
            for k, m in distance_matrices.items():
                logging.debug(
                    "Computing TSNE embedding for outputs of submodule '{}'", k
                )
                t = TSNE(
                    n_components=2,
                    metric="precomputed",
                    init="random",
                    n_jobs=-1,
                )
                t.fit_transform(m)
                e = np.array(t.embedding_)
                e = (e - e.min(axis=0)) / (e.max(axis=0) - e.min(axis=0))
                h.result[k] = e
        tsne_embeddings: dict[str, np.ndarray] = h.result

        # PLOTTING
        h = tb.GuardedBlockHandler(output_dir / "tsne" / "plots.json")
        for _ in h.guard():
            h.result = {}
            for k, e in tsne_embeddings.items():
                logging.debug(
                    "Plotting TSNE embedding for outputs of submodule '{}'", k
                )
                figure = bk.figure(title=k, toolbar_location=None)
                class_scatter(figure, e, y_train.numpy(), "viridis")
                h.result[k] = figure
                export_png(
                    figure, filename=h.output_path.parent / (k + ".png")
                )

    # SEPARABILITY SCORE AND PLOTTING
    if tsne_svc_separability:
        n_classes = len(np.unique(y_train))
        class_idx_pairs = [
            (i, j) for i in range(n_classes) for j in range(i + 1, n_classes)
        ]
        if (n_classes * (n_classes - 1) / 2) > max_class_pairs:
            class_idx_pairs = random.sample(class_idx_pairs, max_class_pairs)
        h = tb.GuardedBlockHandler(output_dir / "svc" / "pairwise_rbf.json")
        # h = tb.GuardedBlockHandler(output_dir / "svc" / "pairwise_linear.json")
        # h = tb.GuardedBlockHandler(output_dir / "svc" / "full_linear.json")
        for _ in h.guard():
            h.result = {}
            # PAIRWISE RBF
            for k, e in tsne_embeddings.items():
                logging.debug("Fitting SVC for outputs of submodule '{}'", k)
                h.result[k] = pairwise_svc_scores(
                    e, y_train, max_class_pairs, kernel="rbf"
                )
                h.result[k] = pairwise_svc_scores(
                    e, y_train, max_class_pairs, kernel="linear"
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

    # PHATE
    if phate:
        # EMBEDDING
        h = tb.GuardedBlockHandler(output_dir / "phate" / "phate.json")
        for _ in h.guard():
            h.result = {}
            for k, z in outputs.items():
                logging.debug(
                    "Computing PHATE embedding for outputs of submodule '{}'",
                    k,
                )
                e = PHATE(verbose=False).fit_transform(z.flatten(1))
                e = (e - e.min(axis=0)) / (e.max(axis=0) - e.min(axis=0))
                h.result[k] = e
        phate_embeddings: dict[str, np.ndarray] = h.result

        # PLOTTING
        h = tb.GuardedBlockHandler(output_dir / "phate" / "plots.json")
        for _ in h.guard():
            h.result = {}
            for k, e in phate_embeddings.items():
                logging.debug(
                    "Plotting PHATE embedding for outputs of submodule '{}'", k
                )
                figure = bk.figure(title=k, toolbar_location=None)
                class_scatter(figure, e, y_train.numpy(), "viridis")
                h.result[k] = figure
                export_png(
                    figure, filename=h.output_path.parent / (k + ".png")
                )


def analyse_training(
    output_dir: str | Path,
    dataset: pl.LightningDataModule | str,
    n_samples: int,
    lv_k: int = 10,
    last_epoch: int | None = None,
    # tsne: bool = True,
    # tsne_svc_separability: bool = True,
):
    """
    For now only plot LV scores per epoch and per submodule

    Args:
        output_path (str | Path): e.g. `./out/resnet18/cifar10/version_1/`
        dataset (pl.LightningDataModule | str):
        n_samples (int): Sorry it's not inferred ¯\\_(ツ)_/¯
        lv_k (int, optional):
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

    if not isinstance(dataset, pl.LightningDataModule):
        dataset = TorchvisionDataset(dataset)
    dataset.setup("fit")
    _, y_train = get_first_n(dataset.train_dataloader(), n_samples)

    data = []
    progress = tqdm(
        enumerate(ckpt_analysis_dirs), desc="Computing LVs", leave=False
    )
    for epoch, p in progress:
        evaluations = tb.load_json(Path(p) / "eval" / "eval.json")
        for sm, z in evaluations.items():
            progress.set_postfix({"epoch": epoch, "submodule": sm})
            v = label_variation(z, y_train, k=lv_k)
            data.append([epoch, sm, float(v)])
    lvs = pd.DataFrame(data, columns=["epoch", "submodule", "lv"])
    lvs.to_csv(output_dir / "lv.csv")

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

    # dfs = []
    # for epoch, p in enumerate(ckpt_analysis_dirs):
    #     with (Path(p) / "svc" / "pairwise_rbf.json").open(
    #         mode="r", encoding="utf-8"
    #     ) as fp:
    #         doc = json.load(fp)  # Prevents loading numpy arrays
    #     data = [
    #         [epoch, k, np.mean([d["score"] for d in v])]
    #         for k, v in doc.items()
    #     ]
    #     df = pd.DataFrame(data, columns=["epoch", "submodule", "mean_score"])
    #     dfs.append(df)
    # tss = pd.concat(dfs, ignore_index=True)
    # tss.to_csv(output_path / "tsne_svc.csv")

    # e = np.linspace(0, last_epoch, num=5, dtype=int)
    # figure = sns.lineplot(
    #     tss[tss["epoch"].isin(e)],
    #     x="submodule",
    #     y="mean_score",
    #     hue="epoch",
    #     size="epoch",
    # )
    # figure.set(title="Separability scores by epoch")
    # figure.set_xticklabels(
    #     figure.get_xticklabels(),
    #     rotation=45,
    #     rotation_mode="anchor",
    #     ha="right",
    # )
    # figure.get_figure().savefig(output_path / "tsne_svc.png")

    # val_acc = metrics["val/acc"].to_numpy()
    # val_loss = metrics["val/loss"].to_numpy()
    # submodules = tss[tss["epoch"] == 0]["submodule"]
    # data = []
    # for s in submodules:
    #     a = tss[tss["submodule"] == s]["mean_score"].to_numpy()
    #     data.append(
    #         [s, np.corrcoef(val_acc, a)[0, 1], np.corrcoef(val_loss, a)[0, 1]],
    #     )
    # correlations = pd.DataFrame(
    #     data, columns=["submodule", "val/acc", "val/loss"]
    # )
    # correlations.to_csv(output_path / "tsne_svc_corr.csv")

    # mcorr = correlations.melt(
    #     id_vars=["submodule"],
    #     var_name="sep. vs.",
    #     value_name="corr.",
    # )
    # grid = sns.FacetGrid(mcorr, col="sep. vs.")
    # grid.map(sns.barplot, "submodule", "corr.")
    # for ax in grid.axes_dict.values():
    #     ax.set_xticklabels(
    #         ax.get_xticklabels(),
    #         rotation=45,
    #         rotation_mode="anchor",
    #         ha="right",
    #     )
    #     ax.set_ylim(-1, 1)
    # grid.fig.savefig(output_path / "tsne_svc_corr.png")


def train_and_analyse_all(
    model: Classifier,
    submodule_names: list[str],
    dataset: pl.LightningDataModule | str,
    output_dir: str | Path,
    model_name: str | None = None,
    n_samples: int = 5000,
    # max_class_pairs: int = 200,
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
        max_class_pairs (int, optional): The last stage of this suite compute
            pairwise separability scores. That means that for each pair of
            distinct classes, a SVC is fitted and a score is computed. If there
            are many classes, the number of pairs can be very large. If this
            number is greater than `n_class_pairs`, then `n_class_pairs` pairs
            are chosen at random for the separability scoring.
    """
    # tb.set_max_nbytes(1000)  # Ensure artefacts
    model_name = model_name or model.__class__.__name__.lower()
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    if isinstance(dataset, str):
        dataset = TorchvisionDataset(dataset)
    _, ckpt = train_model_guarded(
        model,
        dataset,
        output_dir / "model",
        name=model_name,
        max_epochs=512,
        strategy="ddp",
    )
    if model.global_rank == 0:
        version, best_epoch, _ = checkpoint_ves(ckpt)
        p = (
            output_dir
            / "model"
            / "tb_logs"
            / model_name
            / f"version_{version}"
            / "checkpoints"
        )
        for i, ckpt in enumerate(all_ckpt_paths(p)):
            analyse_ckpt(
                model=ckpt,
                model_cls=type(model),
                submodule_names=submodule_names,
                dataset=dataset,
                output_dir=output_dir / f"version_{version}" / str(i),
                n_samples=n_samples,
                tsne=False,
                tsne_svc_separability=False,
                # max_class_pairs=max_class_pairs,
                phate=False,
            )
        analyse_training(
            output_dir / f"version_{version}",
            dataset,
            n_samples=n_samples,
            last_epoch=best_epoch,
        )
