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
    DEFAULT_DATALOADER_KWARGS,
)
from nlnas.classifier import ClusterCorrectionTorchvisionClassifier
from nlnas.logging import setup_logging
from nlnas.training import train_model, train_model_guarded
from nlnas.transforms import EnsuresRGB
from nlnas.utils import dl_targets


def main():
    pl.seed_everything(0)
    dataset_names = [
        # "mnist",
        # "kmnist",
        # "fashionmnist",
        "cifar10",
        "cifar100",
    ]
    transform = tvtr.Compose(
        [
            # tvtr.RandomCrop(32, padding=4),
            # tvtr.RandomHorizontalFlip(),
            tvtr.ToTensor(),
            # tvtr.Normalize(  # Taken from pl_bolts cifar10_normalization
            #     mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            #     std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
            # ),
            tvtr.Resize([64, 64], antialias=True),
            # EnsuresRGB(),
        ]
    )
    for d in dataset_names:
        name = "resnet18_bcc_nn5_b2048_5e-1"
        output_dir = Path("out") / name / d
        ds = TorchvisionDataset(
            d,
            transform=transform,
            dataloader_kwargs={
                "drop_last": True,
                "batch_size": 2048,
                "pin_memory": True,
                "num_workers": 16,
            },
        )
        ds.setup("fit")
        n_classes = len(dl_targets(ds.val_dataloader()))
        image_shape = list(next(iter(ds.val_dataloader()))[0].shape)[1:]
        model = TorchvisionClassifier(
            model_name="alexnet",
            input_shape=image_shape,
            n_classes=n_classes,
            add_final_fc=True,
            sep_submodules=[
                # "model.0.layer1",
                # "model.0.layer2",
                "model.0.layer3",
                "model.0.layer4",
                "model.0.fc",
                "model.1",
            ],
            sep_score="louvain",
            sep_weight=5e-1,
        )
        train_and_analyse_all(
            model=model,
            submodule_names=[
                "model.0.layer1",
                "model.0.layer2",
                "model.0.layer3",
                "model.0.layer4",
                "model.0.fc",
                "model.1",
            ],
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