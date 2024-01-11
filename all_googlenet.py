from itertools import product
from pathlib import Path

import pytorch_lightning as pl
import torchvision.transforms as tvtr
from loguru import logger as logging
from torch import Tensor

from nlnas import (
    TorchvisionClassifier,
    TorchvisionDataset,
    train_and_analyse_all,
)
from nlnas.classifier import TorchvisionClassifier
from nlnas.logging import setup_logging
from nlnas.nlnas import train_and_analyse_all
from nlnas.transforms import cifar10_normalization
from nlnas.tv_dataset import DEFAULT_DATALOADER_KWARGS, TorchvisionDataset


def extract_logits(_module, _inputs, outputs) -> Tensor | None:
    """Googlenet outputs a named tuple instead of a tensor"""
    return outputs.logits if not isinstance(outputs, Tensor) else None


def main():
    pl.seed_everything(0)
    model_names = [
        "googlenet",
    ]
    submodule_names = [
        "model.0.conv1",
        "model.0.conv2",
        "model.0.conv3",
        "model.0.inception3a",
        "model.0.inception4a",
        "model.0.inception4b",
        "model.0.inception4c",
        "model.0.inception4d",
        "model.0.inception5a",
        "model.0.inception5b",
        "model.0.fc",
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
            cifar10_normalization(),
            tvtr.Resize([64, 64], antialias=True),
        ]
    )
    for m, d in product(model_names, dataset_names):
        output_dir = Path("out") / m / d
        dataloader_kwargs = DEFAULT_DATALOADER_KWARGS.copy()
        dataloader_kwargs["batch_size"] = 2048
        datamodule = TorchvisionDataset(
            "cifar10",
            transform=transform,
            dataloader_kwargs=dataloader_kwargs,
        )
        model = TorchvisionClassifier(
            model_name=m,
            n_classes=datamodule.n_classes,
            input_shape=datamodule.image_shape,
        )
        model.model[0].register_forward_hook(extract_logits)
        train_and_analyse_all(
            model=model,
            submodule_names=submodule_names,
            dataset=datamodule,
            output_dir=output_dir,
            model_name=m,
            strategy="ddp_find_unused_parameters_true",
        )


if __name__ == "__main__":
    setup_logging()
    try:
        main()
    except KeyboardInterrupt:
        pass
    except:
        logging.exception(":sad trombone:")
