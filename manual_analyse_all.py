from itertools import product
from pathlib import Path

import pytorch_lightning as pl
import torchvision
from loguru import logger as logging

from nlnas import (
    TorchvisionClassifier,
    TorchvisionDataset,
    VHTorchvisionClassifier,
    train_and_analyse_all,
)
from nlnas.nlnas import analyse_ckpt
from nlnas.training import train_model, train_model_guarded
from nlnas.transforms import EnsuresRGB
from nlnas.utils import all_ckpt_paths, targets


def main():
    pl.seed_everything(0)
    model_name, dataset_name, version = "alexnet", "cifar10", 8
    submodule_names = [
        "model.0.features.0",
        # "model.0.features.3",
        # "model.0.features.6",
        # "model.0.features.8",
        # "model.0.features.10",
        "model.0.features",
        # "model.0.features.9",
        # "model.0.classifier.2",
        # "model.0.classifier.5",
        # "model.0.classifier.6",
        "model.0.classifier",
        "model.1",
    ]
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize([64, 64], antialias=True),
            EnsuresRGB(),
        ]
    )
    dataset = TorchvisionDataset(dataset_name, transform=transform)
    output_dir = Path("out") / (model_name + "_vh") / dataset_name
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
            ckpt,
            submodule_names,
            dataset,
            output_dir / f"version_{version}" / str(i),
            5000,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except:
        logging.exception(":sad trombone:")
