from itertools import product
from pathlib import Path

import pytorch_lightning as pl
import torchvision.transforms as tvtr
from loguru import logger as logging

from nlnas.classifier import TorchvisionClassifier
from nlnas.logging import setup_logging
from nlnas.nlnas import train_and_analyse_all
from nlnas.transforms import *
from nlnas.tv_dataset import DEFAULT_DATALOADER_KWARGS, TorchvisionDataset
from nlnas.utils import best_device


def main():
    pl.seed_everything(0)
    model_names = [
        "resnet18",
        # "resnet34",
        # "resnet50",
        # "resnet101",
        # "resnet152",
    ]
    analysis_submodules = [
        # "model.0.maxpool",
        "model.0.layer1",
        "model.0.layer2",
        "model.0.layer3",
        "model.0.layer4",
        "model.0.fc",
    ]
    datasets = {
        "mnist": tvtr.Compose(
            [
                tvtr.ToTensor(),
                EnsureRGB(),
                mnist_normalization(),
                tvtr.Resize([64, 64], antialias=True),
            ]
        ),
        "fashionmnist": tvtr.Compose(
            [
                tvtr.ToTensor(),
                EnsureRGB(),
                fashionmnist_normalization(),
                tvtr.Resize([64, 64], antialias=True),
            ]
        ),
        "cifar10": tvtr.Compose(
            [
                tvtr.RandomCrop(32, padding=4),
                tvtr.RandomHorizontalFlip(),
                tvtr.ToTensor(),
                cifar10_normalization(),
                tvtr.Resize([64, 64], antialias=True),
            ]
        ),
        "cifar100": tvtr.Compose(
            [
                tvtr.RandomCrop(32, padding=4),
                tvtr.RandomHorizontalFlip(),
                tvtr.ToTensor(),
                cifar10_normalization(),
                tvtr.Resize([64, 64], antialias=True),
            ]
        ),
    }
    for m, (d, t) in product(model_names, datasets.items()):
        try:
            output_dir = Path("out") / m / d
            dataloader_kwargs = DEFAULT_DATALOADER_KWARGS.copy()
            dataloader_kwargs["batch_size"] = 2048
            datamodule = TorchvisionDataset(
                d,
                transform=t,
                dataloader_kwargs=dataloader_kwargs,
            )
            model = TorchvisionClassifier(
                model_name=m,
                n_classes=datamodule.n_classes,
                input_shape=datamodule.image_shape,
            )
            model = model.to(best_device())
            train_model(
                model,
                datamodule,
                output_dir / "model",
                name=m,
                max_epochs=512,
            )
            # train_and_analyse_all(
            #     model=model,
            #     submodule_names=analysis_submodules,
            #     dataset=datamodule,
            #     output_dir=output_dir,
            #     model_name=m,
            # )
        except (KeyboardInterrupt, SystemExit):
            return
        except:
            logging.exception(":sad trombone:")


if __name__ == "__main__":
    setup_logging()
    main()
