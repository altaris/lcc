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

from nlnas.pdist import pdist
from nlnas.plotting import class_scatter
from nlnas.tensor_dataset import TensorDataset
from nlnas.tv_classifier import TorchvisionClassifier
from nlnas.utils import train_model_guarded


def analysis(
    model: TorchvisionClassifier | str | Path,
    submodule_names: str | list[str],
    dataset: TensorDataset | str,
    output_dir: str | Path,
    n_samples: int = 10000,
    max_class_pairs: int = 200,
):
    """
    Full separability analysis and plottings

    Args:
        model (TorchvisionClassifier | str | Path): A model or a path to a
            model checkpoint
        submodule_names (str | list[str]): List or comma-separated list of
            submodule names. For example, the interesting submodules of
            `resnet18` are `maxpool,layer1,layer2,layer3,layer4,fc`
        dataset_name (TensorDataset | str): A tensor dataset or the name of a
            torchvision dataset
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
    if not isinstance(dataset, TensorDataset):
        dataset = TensorDataset.from_torchvision_dataset(dataset)
    assert isinstance(dataset, TensorDataset)

    # MAP SUBMODULE NAMES
    # ... of the model (e.g. `ResNet18`) to those of the actual
    # model (`TorchvisionClassifier`). For example, `layer1` becomes
    # `model.0.layer1`. BUT if the dataset has non-RGB images, an initial 1x1
    # convolution layer is prepended, so the name is actually `model.1.layer1`.
    # In addition to all that, the output layer (`model.1` or `model.2`) is
    # also automatically added to `submodule_names`
    if isinstance(submodule_names, str):
        submodule_names = submodule_names.split(",")
    if dataset.image_shape[0] == 3:
        submodule_names = ["model.0." + s for s in submodule_names]
        submodule_names.append("model.1")
    else:
        submodule_names = ["model.1." + s for s in submodule_names]
        submodule_names.append("model.2")

    # EVALUATION
    logging.debug("Evaluating model")
    h = tb.GuardedBlockHandler(output_dir / "eval" / "eval.json")
    for _ in h.guard():
        h.result = {}
        model.eval()
        model.forward_intermediate(
            dataset.x[:n_samples],
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
    y = dataset.y[:n_samples].numpy()
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
    class_idx_pairs = [
        (i, j)
        for i in range(dataset.n_classes)
        for j in range(i + 1, dataset.n_classes)
    ]
    if (dataset.n_classes * (dataset.n_classes - 1) / 2) > max_class_pairs:
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
        figure.set_xticklabels(figure.get_xticklabels(), rotation=45)
        figure.get_figure().savefig(h.output_path.parent / "separability.png")


def train_and_analyse_all(
    model_name: str,
    submodule_names: str | list[str],
    dataset_name: str,
    output_dir: str | Path,
    n_samples: int = 10000,
    max_class_pairs: int = 200,
):
    """
    Trains a model and performs a separability analysis (see
    `nlnas.nlnas.analyse`) on ALL models, obtained at the end of each epochs.

    Args:
        model_name (str):
        submodule_names (str): List or comma-separated list of submodule names.
            For example, the interesting submodules of `resnet18` are
            `maxpool,layer1,layer2,layer3,layer4,fc`
        dataset_name (str):
        output_dir (str | Path):
        n_samples (int):
        n_class_pairs (int): The last stage of this suite compute pairwise
            separability scores. That means that for each pair of distinct
            classes, a SVC is fitted and a score is computed. If there are many
            classes, the number of pairs can be very large. If this number is
            greater than `n_class_pairs`, then `n_class_pairs` pairs are chosen
            at random for the separability scoring.
    """
    tb.set_max_nbytes(1000)  # Ensure artefacts
    output_dir = Path(output_dir) / model_name / dataset_name
    ds = TensorDataset.from_torchvision_dataset(dataset_name)
    model = TorchvisionClassifier(model_name, ds.image_shape, ds.n_classes)
    train, val = ds.train_test_split_dl()
    train_model_guarded(
        model,
        train,
        val,
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
    all_ckpts = sorted(list(glob(ckpt_glob)))
    for i, ckpt in enumerate(all_ckpts):
        analysis(
            ckpt,
            submodule_names,
            ds,
            output_dir / str(i),
            n_samples,
            max_class_pairs,
        )


def train_and_analyse_best(
    model_name: str,
    submodule_names: str | list[str],
    dataset_name: str,
    output_dir: str | Path,
    n_samples: int = 10000,
    max_class_pairs: int = 200,
):
    """
    Trains a model and performs a separability analysis (see
    `nlnas.nlnas.analyse`) on the best model.

    Args:
        model_name (str):
        submodule_names (str): List or comma-separated list of submodule names.
            For example, the interesting submodules of `resnet18` are
            `maxpool,layer1,layer2,layer3,layer4,fc`
        dataset_name (str):
        output_dir (str | Path):
        n_samples (int):
        n_class_pairs (int): The last stage of this suite compute pairwise
            separability scores. That means that for each pair of distinct
            classes, a SVC is fitted and a score is computed. If there are many
            classes, the number of pairs can be very large. If this number is
            greater than `n_class_pairs`, then `n_class_pairs` pairs are chosen
            at random for the separability scoring.
    """
    tb.set_max_nbytes(1000)  # Ensure artefacts
    output_dir = Path(output_dir) / model_name / dataset_name
    ds = TensorDataset.from_torchvision_dataset(dataset_name)
    model = TorchvisionClassifier(model_name, ds.image_shape, ds.n_classes)
    train, val = ds.train_test_split_dl()
    model = train_model_guarded(
        model,
        train,
        val,
        output_dir / "model",
        name=model_name,
        max_epochs=512,
        strategy="ddp",
    )
    assert isinstance(model, TorchvisionClassifier)
    analysis(
        model, submodule_names, ds, output_dir, n_samples, max_class_pairs
    )
