from pathlib import Path

import bokeh.plotting as bk
import numpy as np
import torchvision
from bokeh.palettes import Category10
from loguru import logger as logging
from sklearn.manifold import TSNE
from torch import Tensor
from turbo_broccoli import GuardedBlockHandler, set_max_nbytes

from nlnas.pdist import pdist
from nlnas.tensor_dataset import TensorDataset
from nlnas.tv_classifier import TorchvisionClassifier
from nlnas.utils import train_model_guarded


def main():
    root_path = Path("export-out/test")

    ds_name = "cifar10"
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Resize(IMAGE_SHAPE[1:]),
            # torchvision.transforms.Normalize([0], [1]),
        ]
    )
    ds = TensorDataset.from_torchvision_dataset(ds_name, transform=transforms)
    train, val = ds.train_test_split_dl()

    model_name = "resnet18"
    model = TorchvisionClassifier(model_name, [3, 32, 32])
    model = train_model_guarded(
        model,
        train,
        val,
        root_path / "model",
        name="resnet18",
        max_epochs=512,
        strategy="ddp",
    )
    assert isinstance(model, TorchvisionClassifier)

    set_max_nbytes(1000)

    n = 1000
    submodules = [
        "model.0.maxpool",
        "model.0.layer1",
        "model.0.layer2",
        "model.0.layer3",
        "model.0.layer4",
        "model.0.avgpool",
        "model.0.fc",
    ]
    h = GuardedBlockHandler(root_path / "eval/eval.json", name="eval")
    for _ in h.guard():
        h.result = {}
        model.eval()
        model.forward_intermediate(
            ds.x[:n],
            submodules,
            h.result,
        )
    intermediate_outs: dict[str, Tensor] = h.result

    h = GuardedBlockHandler(root_path / "pdist", name="pdist")
    for k in h.guard(submodules):
        h.result[k] = pdist(
            intermediate_outs[k].numpy(),
            chunk_size=int(n / 10),
            chunk_path=root_path / "pdist/chunks",
        )
    distance_matrices: dict[str, np.ndarray] = h.result

    h = GuardedBlockHandler(root_path / "tsne", name="tsne")
    for k in h.guard(submodules):
        logging.debug(
            "Computing TSNE embedding for outputs of submodule '{}'", k
        )
        t = TSNE(
            n_components=2, metric="precomputed", init="random", n_jobs=-1
        )
        t.fit_transform(distance_matrices[k])
        h.result[k] = t.embedding_
    embeddings: dict[str, np.ndarray] = h.result

    h = GuardedBlockHandler(root_path / "plots", name="plots")
    for k in h.guard(submodules):
        logging.debug(
            "Plotting TSNE embedding for outputs of submodule '{}'", k
        )
        x, y = embeddings[k], ds.y[:n].numpy()
        figure, palette = bk.figure(), Category10[10]
        figure.title = k
        for j in range(10):
            a, b = x[y == j], palette[j]
            figure.scatter(
                a[:, 0],
                a[:, 1],
                color=b,
                legend_label=str(j),
                line_width=0,
                size=3,
            )
        h.result[k] = figure


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except:
        logging.exception(":sad trombone:")
