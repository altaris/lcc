from itertools import product
from pathlib import Path

import pytorch_lightning as pl
from _parameters import DATASETS
from loguru import logger as logging

from nlnas import (
    DEFAULT_DATALOADER_KWARGS,
    TorchvisionClassifier,
    TorchvisionDataset,
    best_device,
    train_and_analyse_all,
    train_model_guarded,
)
from nlnas.logging import setup_logging


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
        # "model.maxpool",
        "model.layer1",
        "model.layer2",
        "model.layer3",
        "model.layer4",
        "model.fc",
    ]
    for m, (d, t) in product(model_names, DATASETS.items()):
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
            train_model_guarded(
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