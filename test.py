from pathlib import Path

import bokeh.plotting as bk
import numpy as np
from sklearn import mixture
import torchvision
import turbo_broccoli as tb
from bokeh.palettes import viridis
from loguru import logger as logging
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from torch import Tensor

from nlnas.pdist import pdist
from nlnas.plotting import gaussian_mixture_plot
from nlnas.tensor_dataset import TensorDataset
from nlnas.tv_classifier import TorchvisionClassifier
from nlnas.utils import train_model_guarded


def main():
    model_name, ds_name = "resnet50", "cifar10"
    input_shape, n_classes = [3, 32, 32], 10
    n = 10000  # Number of samples for eval

    root_path = Path(f"export-out/{model_name}/{ds_name}")

    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Resize(IMAGE_SHAPE[1:]),
            torchvision.transforms.Normalize([0], [1]),
        ]
    )
    ds = TensorDataset.from_torchvision_dataset(ds_name, transform=transforms)
    train, val = ds.train_test_split_dl()

    model = TorchvisionClassifier(model_name, input_shape)
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

    submodules = [
        "model.0.maxpool",
        "model.0.layer1",
        "model.0.layer2",
        "model.0.layer3",
        "model.0.layer4",
        "model.0.fc",
        "model.1",
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
            h.result[k] = t.embedding_
    embeddings: dict[str, np.ndarray] = h.result

    h = tb.GuardedBlockHandler(root_path / "gm" / "gm.json")
    for _ in h.guard():
        h.result = {}
        for k, e in embeddings.items():
            logging.debug("Fitting GM for outputs of submodule '{}'", k)
            h.result[k] = GaussianMixture(n_classes).fit(e)
    mixtures: dict[str, GaussianMixture] = h.result

    h = tb.GuardedBlockHandler(root_path / "plots" / "plots.json")
    for _ in h.guard():
        h.result = {}
        for k, e in embeddings.items():
            logging.debug(
                "Plotting TSNE embedding for outputs of submodule '{}'", k
            )
            y = ds.y[:n].numpy()
            figure, palette = bk.figure(), viridis(n_classes)
            figure.title = f"{model_name}/{k}, {ds_name}"
            for j in range(10):
                a, b = e[y == j], palette[j]
                figure.scatter(
                    a[:, 0],
                    a[:, 1],
                    color=b,
                    legend_label=str(j),
                    line_width=0,
                    size=3,
                )
            gaussian_mixture_plot(
                figure,
                mixtures[k],
                x_min=e[:, 0].min() - 10,
                x_max=e[:, 0].max() + 10,
                y_min=e[:, 1].min() - 10,
                y_max=e[:, 1].max() + 10,
            )
            h.result[k] = figure


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except:
        logging.exception(":sad trombone:")
