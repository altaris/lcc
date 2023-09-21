from pathlib import Path

import bokeh.plotting as bk
import numpy as np
import pandas as pd
import seaborn as sns
import torchvision
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


def main():
    model_name, ds_name = "resnet18", "cifar10"
    n = 10000  # Number of samples for eval

    root_path = Path(f"export-out/{model_name}/{ds_name}")

    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Resize([256, 256], antialias=True),
            torchvision.transforms.Normalize([0], [1]),
        ]
    )
    ds = TensorDataset.from_torchvision_dataset(ds_name, transform=transforms)
    input_shape = list(ds.x[0].shape)
    n_classes = len(ds.y[0]) if ds.y.ndim == 2 else len(ds.y.unique())

    train, val = ds.train_test_split_dl()
    model = TorchvisionClassifier(model_name, input_shape, n_classes)
    model = train_model_guarded(
        model,
        train,
        val,
        root_path / "model",
        name=model_name,
        max_epochs=512,
        strategy="ddp",
    )
    assert isinstance(model, TorchvisionClassifier)
    model.eval()

    tb.set_max_nbytes(1000)  # Ensure artefacts

    # submodules = [  # googlenet
    #     "model.0.maxpool1",
    #     "model.0.maxpool2",
    #     "model.0.inception3a",
    #     "model.0.maxpool3",
    #     "model.0.inception4a",
    #     "model.0.inception4b",
    #     "model.0.inception4c",
    #     "model.0.inception4d",
    #     "model.0.maxpool4",
    #     "model.0.inception5a",
    #     "model.0.inception5b",
    #     "model.0.aux1",
    #     "model.0.aux2",
    #     "model.0.fc",
    #     "model.1",
    # ]
    # submodules = [  # resnet, 3 chans
    #     "model.0.maxpool",
    #     "model.0.layer1",
    #     "model.0.layer2",
    #     "model.0.layer3",
    #     "model.0.layer4",
    #     "model.0.fc",
    #     "model.1",
    # ]
    submodules = [  # resnet, 1 chan
        "model.1.maxpool",
        "model.1.layer1",
        "model.1.layer2",
        "model.1.layer3",
        "model.1.layer4",
        "model.1.fc",
        "model.2",
    ]

    h = tb.GuardedBlockHandler(root_path / "eval" / "eval.json")
    for _ in h.guard():
        h.result = {}
        model.eval()
        model.forward_intermediate(
            ds.x[:n],
            submodules,
            h.result,
        )
    intermediate_outs: dict[str, Tensor] = h.result

    h = tb.GuardedBlockHandler(root_path / "pdist" / "pdist.json")
    for _ in h.guard():
        h.result = {}
        for k, z in intermediate_outs.items():
            z = z.flatten(1).numpy()
            h.result[k] = pdist(
                z,
                chunk_size=int(n / 10),
                chunk_path=h.output_path.parent / "chunks" / k,
            )
    distance_matrices: dict[str, np.ndarray] = h.result

    h = tb.GuardedBlockHandler(root_path / "tsne" / "tsne.json")
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

    # h = tb.GuardedBlockHandler(root_path / "gm" / "gm.json")
    # for _ in h.guard():
    #     h.result = {}
    #     for k, e in embeddings.items():
    #         logging.debug("Fitting GM for outputs of submodule '{}'", k)
    #         h.result[k] = GaussianMixture(n_classes).fit(e)
    # mixtures: dict[str, GaussianMixture] = h.result

    h = tb.GuardedBlockHandler(root_path / "tsne" / "plots.json")
    for _ in h.guard():
        h.result = {}
        for k, e in embeddings.items():
            logging.debug(
                "Plotting TSNE embedding for outputs of submodule '{}'", k
            )
            figure = bk.figure(
                title=f"{model_name}/{k}, {ds_name}",
                toolbar_location=None,
            )
            class_scatter(figure, e, y, "viridis")
            h.result[k] = figure
            export_png(figure, filename=h.output_path.parent / (k + ".png"))

    class_idx_pairs = [
        (i, j) for i in range(n_classes) for j in range(i + 1, n_classes)
    ]
    y = ds.y[:n].numpy()
    h = tb.GuardedBlockHandler(root_path / "svc" / "svc.json")
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

        scores = [[d["score"] for d in v] for v in h.result.values()]
        df = pd.DataFrame(scores, index=h.result.keys())
        df = df.reset_index(names="submodule")
        df = df.melt(id_vars=["submodule"], value_name="score")

        figure = sns.lineplot(df, x="submodule", y="score")
        figure.set(title="Pairwise SVC separability scores")
        figure.set_xticklabels(figure.get_xticklabels(), rotation=45)
        figure.get_figure().savefig(h.output_path.parent / "separability.png")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except:
        logging.exception(":sad trombone:")
