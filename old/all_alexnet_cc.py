from itertools import product
from pathlib import Path

import pytorch_lightning as pl
import torchvision.transforms as tvtr
from loguru import logger as logging

from nlnas.classifier import TorchvisionClassifier
from nlnas.logging import setup_logging
from nlnas.nlnas import train_and_analyse_all
from nlnas.training import best_checkpoint_path, train_model_guarded
from nlnas.transforms import cifar10_normalization
from nlnas.dataset import DEFAULT_DATALOADER_KWARGS, TorchvisionDataset
from nlnas.utils import best_device


def main():
    pl.seed_everything(0)
    analysis_submodules = [
        "model.features.0",
        # "model.features.3",
        "model.features.6",
        # "model.features.8",
        "model.features.10",
        # "model.features",
        "model.classifier.1",
        "model.classifier.4",
        "model.classifier.6",
        # "model.classifier",
        # "model"
    ]
    cor_submodules = [
        "model.classifier.1",
        "model.classifier.4",
        "model.classifier.6",
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
    weight_exponents = [0, 1, 2, 3, 5, 10]
    batch_sizes = [2048]
    for we, bs in product(weight_exponents, batch_sizes):
        try:
            exp_name = f"alexnet_l5_b{bs}_1e-{we}"
            output_dir = Path("out") / exp_name / "cifar10"
            dataloader_kwargs = DEFAULT_DATALOADER_KWARGS.copy()
            dataloader_kwargs["batch_size"] = 2048
            datamodule = TorchvisionDataset(
                "cifar10",
                transform=transform,
                dataloader_kwargs=dataloader_kwargs,
            )
            model = TorchvisionClassifier(
                model_name="alexnet",
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
