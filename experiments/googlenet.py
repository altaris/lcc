from itertools import product
from pathlib import Path

import pytorch_lightning as pl
import torchvision.transforms as tvtr
from loguru import logger as logging
from torch import Tensor

from nlnas.classifier import TorchvisionClassifier
from nlnas.logging import setup_logging
from nlnas.nlnas import train_and_analyse_all
from nlnas.transforms import *
from nlnas.tv_dataset import DEFAULT_DATALOADER_KWARGS, TorchvisionDataset
from nlnas.utils import best_device


def extract_logits(_module, _inputs, outputs) -> Tensor | None:
    """Googlenet outputs a named tuple instead of a tensor"""
    return outputs.logits if not isinstance(outputs, Tensor) else None


def main():
    pl.seed_everything(0)
    model_names = ["googlenet"]
    analysis_submodules = [
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
            model.model[0].register_forward_hook(extract_logits)
            train_model_guarded(
                model,
                datamodule,
                output_dir / "model",
                name=m,
                max_epochs=512,
                strategy="ddp_find_unused_parameters_true",
            )
            # train_and_analyse_all(
            #     model=model,
            #     submodule_names=analysis_submodules,
            #     dataset=datamodule,
            #     output_dir=output_dir,
            #     model_name=m,
            #     strategy="ddp_find_unused_parameters_true",
            # )
        except (KeyboardInterrupt, SystemExit):
            pass
        except:
            logging.exception(":sad trombone:")


if __name__ == "__main__":
    setup_logging()
    main()
