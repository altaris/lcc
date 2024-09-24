from itertools import product
from pathlib import Path

import pytorch_lightning as pl
import torchvision.transforms as tvtr

from nlnas.classifier import TorchvisionClassifier
from nlnas.dataset import DEFAULT_DATALOADER_KWARGS, TorchvisionDataset
from nlnas.logging import setup_logging
from nlnas.nlnas import train_and_analyse_all
from nlnas.transforms import cifar10_normalization
from nlnas.utils import best_device


def main():
    pl.seed_everything(0)
    model_names = ["vit_b_16"]
    analysis_submodules = [
        # "model.encoder.layers.encoder_layer_0.mlp",
        # "model.encoder.layers.encoder_layer_1.mlp",
        "model.encoder.layers.encoder_layer_2.mlp",
        # "model.encoder.layers.encoder_layer_3.mlp",
        # "model.encoder.layers.encoder_layer_4.mlp",
        # "model.encoder.layers.encoder_layer_5.mlp",
        "model.encoder.layers.encoder_layer_6.mlp",
        # "model.encoder.layers.encoder_layer_7.mlp",
        # "model.encoder.layers.encoder_layer_8.mlp",
        # "model.encoder.layers.encoder_layer_9.mlp",
        # "model.encoder.layers.encoder_layer_10.mlp",
        "model.encoder.layers.encoder_layer_11.mlp",
        "model.heads",
    ]
    dataset_names = [
        "cifar10",
        "cifar100",
    ]
    transform = tvtr.Compose(
        [
            tvtr.RandomCrop(32, padding=4),
            tvtr.RandomHorizontalFlip(),
            tvtr.ToTensor(),
            cifar10_normalization(),
            tvtr.Resize([224, 224], antialias=True),
            # EnsuresRGB(),
        ]
    )
    for m, d in product(model_names, dataset_names):
        try:
            output_dir = Path("out") / m / d
            dataloader_kwargs = DEFAULT_DATALOADER_KWARGS.copy()
            dataloader_kwargs["batch_size"] = 512
            datamodule = TorchvisionDataset(
                d,
                transform=transform,
                dataloader_kwargs=dataloader_kwargs,
            )
            model = TorchvisionClassifier(
                model_name=m,
                n_classes=datamodule.n_classes,
                input_shape=datamodule.image_shape,
            )
            model = model.to(best_device())
            train_and_analyse_all(
                model=model,
                submodule_names=analysis_submodules,
                dataset=datamodule,
                output_dir=output_dir,
                model_name=m,
                n_samples=1000,
            )
        except (KeyboardInterrupt, SystemExit):
            return
        # except:
        #     logging.exception(":sad trombone:")


if __name__ == "__main__":
    setup_logging()
    main()
