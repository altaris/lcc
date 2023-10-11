from itertools import product
from pathlib import Path

import pytorch_lightning as pl
import torch
import torchvision
import torchvision.transforms as tvtr
from loguru import logger as logging

from nlnas import (
    TorchvisionClassifier,
    TorchvisionDataset,
    VHTorchvisionClassifier,
    train_and_analyse_all,
)
from nlnas.training import train_model, train_model_guarded
from nlnas.transforms import EnsuresRGB
from nlnas.utils import targets


def main():
    pl.seed_everything(0)
    model_names = [
        "alexnet",
    ]
    submodule_names = [
        # "model.0.features.0",
        # "model.0.features.3",
        # "model.0.features.6",
        # "model.0.features.8",
        # "model.0.features.10",
        # "model.0.features",
        # "model.0.classifier.1",
        # "model.0.classifier.4",
        # "model.0.classifier.6",
        "model.0.classifier",
        "model.1",
        # "model"
    ]
    dataset_names = [
        # "mnist",
        # "kmnist",
        # "fashionmnist",
        "cifar10",
        # "cifar100",
    ]
    transform = tvtr.Compose(
        [
            tvtr.RandomCrop(32, padding=4),
            tvtr.RandomHorizontalFlip(),
            tvtr.ToTensor(),
            tvtr.Normalize(  # Taken from pl_bolts cifar10_normalization
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
            ),
            tvtr.Resize([64, 64], antialias=True),
            # EnsuresRGB(),
        ]
    )
    for model_name, dataset_name in product(model_names, dataset_names):
        output_dir = Path("out") / (model_name + "_vh") / dataset_name
        dataset = TorchvisionDataset(dataset_name, transform=transform)
        dataset.setup("fit")
        n_classes = len(targets(dataset.val_dataloader()))
        image_shape = list(next(iter(dataset.val_dataloader()))[0].shape)[1:]
        model = VHTorchvisionClassifier(
            model_name=model_name,
            n_classes=n_classes,
            vh_submodules=submodule_names,
            add_final_fc=True,
            input_shape=image_shape,
            horizontal_lr=1e-2,
        )
        # train_model_guarded(
        train_model(
            model,
            dataset,
            output_dir / "model",
            name=model_name,
            max_epochs=512,
            strategy="ddp",
        )
        # train_and_analyse_all(
        #     model=model,
        #     submodule_names=submodule_names,
        #     dataset=ds,
        #     output_dir=output_dir,
        #     model_name=m + "_vh",
        # )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except:
        logging.exception(":sad trombone:")
