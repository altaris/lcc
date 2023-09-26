"""Main module"""

from glob import glob
import random
from pathlib import Path

import bokeh.plotting as bk
import numpy as np
import pandas as pd
import seaborn as sns
import turbo_broccoli as tb
from bokeh.io import export_png
from loguru import logger as logging
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from torch import Tensor
import pytorch_lightning as pl

from nlnas.tv_dataset import TorchvisionDataset
from nlnas.utils import get_first_n


from .pdist import pdist
from .plotting import class_scatter
from .classifier import TorchvisionClassifier
from .training import train_model_guarded


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
        model = TorchvisionClassifier.load_from_checkpoint(model)
    assert isinstance(model, TorchvisionClassifier)
    model.eval()

    # LOAD DATASET IF NEEDED
    if not isinstance(dataset, pl.LightningDataModule):
        dataset = TorchvisionDataset(dataset)
    dataset.setup("fit")

    # EVALUATION
    logging.debug("Evaluating model")
    h = tb.GuardedBlockHandler(output_dir / "eval" / "eval.json")
    for _ in h.guard():
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
    for _ in h.guard():
        h.result = {}
        for k, z in outputs.items():
            logging.debug(
                "Computing distance matrix for outputs of submodule '{}'", k
            )
            h.result[k] = pdist(
                z.flatten(1).numpy(),
                chunk_size=int(n_samples / 10),
                chunk_path=h.output_path.parent / "chunks" / k,
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
            figure = bk.figure(
                title=k,
                toolbar_location=None,
            )
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
    h = tb.GuardedBlockHandler(output_dir / "svc" / "svc.json")
    for _ in h.guard():
        h.result = {}
        for k, e in embeddings.items():
            logging.debug("Fitting SVC for outputs of submodule '{}'", k)
            h.result[k] = []
            for i, j in class_idx_pairs:
                yij = (y == i) + (y == j)
                a, b = e[yij], y[yij] == i
                svc = SVC(kernel="rbf").fit(a, b)
                h.result[k].append(
                    {"idx": (i, j), "svc": svc, "score": svc.score(a, b)}
                )

        # Plotting is done here to be in the guarded block
        logging.debug("Plotting separability scores")
        scores = [np.mean([d["score"] for d in v]) for v in h.result.values()]
        df = pd.DataFrame(
            scores, index=h.result.keys(), columns=["mean_score"]
        )
        df = df.reset_index(names="submodule")
        figure = sns.lineplot(df, x="submodule", y="mean_score")
        figure.set(title="Mean pairwise separability score")
        figure.set_xticklabels(
            figure.get_xticklabels(),
            rotation=45,
            rotation_mode="anchor",
            ha="right",
        )
        figure.get_figure().savefig(h.output_path.parent / "separability.png")


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
    tb.set_max_nbytes(1000)  # Ensure artefacts
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
    ckpt_glob = str(
        output_dir
        / "model"
        / "tb_logs"
        / model_name
        / "version_0"
        / "checkpoints"
        / "*.ckpt"
    )
    for i, ckpt in enumerate(glob(ckpt_glob)):
        analysis(
            ckpt,
            submodule_names,
            dataset,
            output_dir / str(i),
            n_samples,
            max_class_pairs,
        )
