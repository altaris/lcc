from itertools import product
from pathlib import Path

import pytorch_lightning as pl
import torchvision.transforms as tvtr
from loguru import logger as logging

from nlnas.classifier import TorchvisionClassifier
from nlnas.dataset import DEFAULT_DATALOADER_KWARGS, TorchvisionDataset
from nlnas.logging import setup_logging
from nlnas.training import train_model_guarded
from nlnas.transforms import cifar10_normalization
from nlnas.utils import best_device


def main():
    pl.seed_everything(0)
    analysis_submodules = [
        "model.layer1",
        "model.layer2",
        "model.layer3",
        "model.layer4",
        "model.fc",
    ]
    cor_submodules = [
        # "model.layer1",
        # "model.layer2",
        # "model.layer3",
        "model.layer4",
        "model.fc",
    ]
    transform = tvtr.Compose(
        [
            tvtr.RandomCrop(32, padding=4),
            tvtr.RandomHorizontalFlip(),
            tvtr.ToTensor(),
            cifar10_normalization(),
            tvtr.Resize([64, 64], antialias=True),
            # EnsuresRGB(),
        ]
    )
    weight_exponents = [1, 2, 3, 4, 5]
    batch_sizes = [2048]
    for we, bs in product(weight_exponents, batch_sizes):
        try:
            exp_name = f"resnet18_l5_b{bs}_1e-{we}"
            output_dir = Path("out") / exp_name / "cifar10"
            dataloader_kwargs = DEFAULT_DATALOADER_KWARGS.copy()
            dataloader_kwargs["batch_size"] = 2048
            datamodule = TorchvisionDataset(
                "cifar10",
                transform=transform,
                dataloader_kwargs=dataloader_kwargs,
            )
            model = TorchvisionClassifier(
                model_name="resnet18",
                input_shape=datamodule.image_shape,
                n_classes=datamodule.n_classes,
                cor_submodules=cor_submodules,
                cor_type="louvain",
                cor_weight=10 ** (-we),
            )
            model = model.to(best_device())
            train_model_guarded(
                model,
                datamodule,
                output_dir / "model",
                name=exp_name,
                max_epochs=512,
            )
            # train_and_analyse_all(
            #     model=model,
            #     submodule_names=analysis_submodules,
            #     dataset=datamodule,
            #     output_dir=output_dir,
            #     model_name=name,
            # )
        except (KeyboardInterrupt, SystemExit):
            break
        except:
            logging.exception(":sad trombone:")


if __name__ == "__main__":
    setup_logging()
    main()
