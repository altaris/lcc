"""Main module"""

import random
from glob import glob
from pathlib import Path

import bokeh.plotting as bk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import turbo_broccoli as tb
from bokeh.io import export_png
from loguru import logger as logging
from sklearn.manifold import TSNE
from torch import Tensor

from .separability import pairwise_svc_scores
from .classifier import TorchvisionClassifier
from .pdist import pdist
from .plotting import class_scatter
from .training import train_model_guarded
from .tv_dataset import TorchvisionDataset
from .utils import get_first_n


def analysis(
    model: TorchvisionClassifier | str | Path,
    submodule_names: list[str],
    dataset: pl.LightningDataModule | str,
    output_dir: str | Path,
    n_samples: int = 5000,
    max_class_pairs: int = 200,
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
    """
    output_dir = Path(output_dir)

    # LOAD MODEL IF NEEDED
    if not isinstance(model, TorchvisionClassifier):
        logging.info("Analysing checkpoint {}", str(model))
        model = TorchvisionClassifier.load_from_checkpoint(model)
    assert isinstance(model, TorchvisionClassifier)
    model.eval()

    # LOAD DATASET IF NEEDED
    if not isinstance(dataset, pl.LightningDataModule):
        dataset = TorchvisionDataset(dataset)
    dataset.setup("fit")

    # EVALUATION
    h = tb.GuardedBlockHandler(output_dir / "eval" / "eval.json")
    for _ in h.guard():
        logging.debug("Evaluating model")
        h.result = {}
        model.eval()
        model.forward_intermediate(
            get_first_n(dataset.train_dataloader(), n_samples)[0],
            submodule_names,
            h.result,
        )
    outputs: dict[str, Tensor] = h.result

    # COMPUTE DISTANCE MATRICES
    h = tb.GuardedBlockHandler(output_dir / "pdist" / "pdist.json")
    chunk_path = h.output_path.parent / "chunks"
    for _ in h.guard():
        h.result = {}
        for k, z in outputs.items():
            logging.debug(
                "Computing distance matrix for outputs of submodule '{}'", k
            )
            h.result[k] = pdist(
                z.flatten(1).numpy(),
                chunk_size=int(n_samples / 10),
                chunk_path=chunk_path / k,
            )
    if chunk_path.is_dir():
        logging.debug("Removing pdist chuncks directory '{}'", chunk_path)
        try:
            chunk_path.rmdir()
        except OSError as err:
            logging.error(
                "Could not remove chunks directory '{}': {}", chunk_path, err
            )
    distance_matrices: dict[str, np.ndarray] = h.result

    # TSNE EMBEDDING
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
    embeddings: dict[str, np.ndarray] = h.result

    # TSNE PLOTTING
    y = get_first_n(dataset.train_dataloader(), n_samples)[1].numpy()
    h = tb.GuardedBlockHandler(output_dir / "tsne" / "plots.json")
    for _ in h.guard():
        h.result = {}
        for k, e in embeddings.items():
            logging.debug(
                "Plotting TSNE embedding for outputs of submodule '{}'", k
            )
            figure = bk.figure(title=k, toolbar_location=None)
            class_scatter(figure, e, y, "viridis")
            h.result[k] = figure
            export_png(figure, filename=h.output_path.parent / (k + ".png"))

    # SEPARABILITY SCORE AND PLOTTING
    n_classes = len(np.unique(y))
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
        for k, e in embeddings.items():
            logging.debug("Fitting SVC for outputs of submodule '{}'", k)
            h.result[k] = pairwise_svc_scores(
                e, y, max_class_pairs, kernel="rbf"
            )
            h.result[k] = pairwise_svc_scores(
                e, y, max_class_pairs, kernel="linear"
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
            [k, np.mean([d["score"] for d in v])] for k, v in h.result.items()
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
        figure.get_figure().savefig(h.output_path.parent / "separability.png")
        plt.clf()


def train_and_analyse_all(
    model: TorchvisionClassifier,
    submodule_names: list[str],
    dataset: pl.LightningDataModule | str,
    output_dir: str | Path,
    model_name: str | None = None,
    n_samples: int = 5000,
    max_class_pairs: int = 200,
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
    train_model_guarded(
        model,
        dataset,
        output_dir / "model",
        name=model_name,
        max_epochs=512,
        strategy="ddp",
    )
    if model.global_rank == 0:
        p = (
            output_dir
            / "model"
            / "tb_logs"
            / model_name
            / "version_0"
            / "checkpoints"
        )
        for i, ckpt in enumerate(all_ckpt_paths(p)):
            analysis(
                ckpt,
                submodule_names,
                dataset,
                output_dir / str(i),
                n_samples,
                max_class_pairs,
            )
