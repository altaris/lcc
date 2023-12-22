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
    train_and_analyse_all,
)
from nlnas.classifier import ClusterCorrectionTorchvisionClassifier
from nlnas.logging import setup_logging
from nlnas.training import train_model, train_model_guarded
from nlnas.transforms import EnsuresRGB
from nlnas.utils import dl_targets


def main():
    pl.seed_everything(0)
    submodule_names = [
        "model.0.features.0",
        "model.0.features.3",
        "model.0.features.6",
        "model.0.features.8",
        "model.0.features.10",
        # "model.0.features",
        "model.0.classifier.1",
        "model.0.classifier.4",
        "model.0.classifier.6",
        # "model.0.classifier",
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
    for d in dataset_names:
        name = "alexnet_bcc_1e-4"
        output_dir = Path("out") / name / d
        ds = TorchvisionDataset(d, transform=transform)
        ds.setup("fit")
        n_classes = len(dl_targets(ds.val_dataloader()))
        image_shape = list(next(iter(ds.val_dataloader()))[0].shape)[1:]
        model = ClusterCorrectionTorchvisionClassifier(
            model_name="alexnet",
            n_classes=n_classes,
            add_final_fc=True,
            input_shape=image_shape,
            sep_submodules=[
                "model.0.classifier.1",
                "model.0.classifier.4",
                "model.0.classifier.6",
            ],
        )
        # train_model(
        #     model,
        #     ds,
        #     output_dir / "model",
        #     name=m,
        #     max_epochs=512,
        #     # strategy="ddp_find_unused_parameters_true",
        # )
        train_and_analyse_all(
            model=model,
            submodule_names=submodule_names,
            dataset=ds,
            output_dir=output_dir,
            model_name=name,
        )


if __name__ == "__main__":
    setup_logging()
    try:
        main()
    except KeyboardInterrupt:
        pass
    except:
        logging.exception(":sad trombone:")
