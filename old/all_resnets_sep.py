from itertools import product
from pathlib import Path

import pytorch_lightning as pl
import torchvision
import torchvision.transforms as tvtr
from loguru import logger as logging

from nlnas import (
    TorchvisionClassifier,
    TorchvisionDataset,
    train_and_analyse_all,
)
from nlnas.classifier import TruncatedClassifier
from nlnas.logging import setup_logging
from nlnas.training import train_model, train_model_guarded
from nlnas.utils import dl_targets


def main():
    pl.seed_everything(0)
    backbones = [
        "resnet18",
    ]
    separation_score = "ggd"
    obs_submodules = [
        # "model.0.layer1.0"
        # "model.0.layer1.1"
        "model.0.layer1",
        # "model.0.layer2.0"
        # "model.0.layer2.1"
        "model.0.layer2",
        # "model.0.layer3.0"
        # "model.0.layer3.1"
        "model.0.layer3",
        # "model.0.layer4.0"
        # "model.0.layer4.1"
        "model.0.layer4",
        "model.0.fc",
        "model.1",
    ]
    sep_submodules = [
        # "model.0.layer1.0"
        # "model.0.layer1.1"
        # "model.0.layer1",
        # "model.0.layer2.0"
        # "model.0.layer2.1"
        # "model.0.layer2",
        # "model.0.layer3.0"
        # "model.0.layer3.1"
        # "model.0.layer3",
        # "model.0.layer4.0"
        # "model.0.layer4.1"
        "model.0.layer4",
        # "model.0.fc",
        # "model.1",
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
        ]
    )
    for backbone, dataset_name in product(backbones, dataset_names):
        model_name = backbone + "_" + separation_score + "_l4_w1"
        output_dir = Path("out") / model_name / dataset_name
        dataset = TorchvisionDataset(
            dataset_name,
            transform=transform,
            dataloader_kwargs={
                "batch_size": 256,
                "pin_memory": True,
                "num_workers": 16,
            },
        )
        dataset.setup("fit")
        n_classes = len(dl_targets(dataset.val_dataloader()))
        image_shape = list(next(iter(dataset.val_dataloader()))[0].shape)[1:]
        model = TorchvisionClassifier(
            model_name=backbone,
            n_classes=n_classes,
            input_shape=image_shape,
            add_final_fc=True,
            sep_submodules=sep_submodules,
            sep_score=separation_score,
            sep_weight=1,
        )
        train_and_analyse_all(
            model=model,
            submodule_names=obs_submodules,
            dataset=dataset,
            output_dir=output_dir,
            model_name=model_name,
        )


if __name__ == "__main__":
    setup_logging()
    try:
        main()
    except KeyboardInterrupt:
        pass
    except:
        logging.exception(":sad trombone:")
