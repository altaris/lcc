from itertools import product
from pathlib import Path

import pytorch_lightning as pl
import torchvision.transforms as tvtr

from nlnas.classifier import TorchvisionClassifier
from nlnas.logging import setup_logging
from nlnas.nlnas import train_and_analyse_all
from nlnas.transforms import cifar10_normalization
from nlnas.tv_dataset import DEFAULT_DATALOADER_KWARGS, TorchvisionDataset
from nlnas.utils import best_device


def main():
    pl.seed_everything(0)
    model_names = ["vit_b_16"]
    analysis_submodules = [
        "model.0.encoder.layers.encoder_layer_0",
        "model.0.encoder.layers.encoder_layer_1",
        "model.0.encoder.layers.encoder_layer_2",
        "model.0.encoder.layers.encoder_layer_3",
        "model.0.encoder.layers.encoder_layer_4",
        "model.0.encoder.layers.encoder_layer_5",
        "model.0.encoder.layers.encoder_layer_6",
        "model.0.encoder.layers.encoder_layer_7",
        "model.0.encoder.layers.encoder_layer_8",
        "model.0.encoder.layers.encoder_layer_9",
        "model.0.encoder.layers.encoder_layer_10",
        "model.0.encoder.layers.encoder_layer_11",
        "model.0.heads",
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
            dataloader_kwargs["pin_memory"] = False
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
        except KeyboardInterrupt:
            return
        # except:
        #     logging.exception(":sad trombone:")


if __name__ == "__main__":
    setup_logging()
    main()
    # try:
    #     main()
    # except KeyboardInterrupt:
    #     pass
    # except:
    #     logging.exception(":sad trombone:")
