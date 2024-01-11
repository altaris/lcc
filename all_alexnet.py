from pathlib import Path

import pytorch_lightning as pl
import torchvision.transforms as tvtr
from loguru import logger as logging

from nlnas.classifier import TorchvisionClassifier
from nlnas.logging import setup_logging
from nlnas.nlnas import train_and_analyse_all
from nlnas.training import best_checkpoint_path, train_model_guarded
from nlnas.tv_dataset import DEFAULT_DATALOADER_KWARGS, TorchvisionDataset


def main():
    pl.seed_everything(0)
    analysis_submodules = [
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
    ]
    output_dir = Path("out") / "alexnet" / "cifar10"
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
    dataloader_kwargs = DEFAULT_DATALOADER_KWARGS.copy()
    dataloader_kwargs["batch_size"] = 2048
    datamodule = TorchvisionDataset(
        "cifar10",
        transform=transform,
        dataloader_kwargs=dataloader_kwargs,
    )
    model = TorchvisionClassifier(
        model_name="alexnet",
        n_classes=datamodule.n_classes,
        input_shape=datamodule.image_shape,
    )
    train_and_analyse_all(
        model=model,
        submodule_names=analysis_submodules,
        dataset=datamodule,
        output_dir=output_dir,
        model_name="alexnet",
    )


if __name__ == "__main__":
    setup_logging()
    try:
        main()
    except KeyboardInterrupt:
        pass
    except:
        logging.exception(":sad trombone:")
