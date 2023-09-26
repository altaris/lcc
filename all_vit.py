from itertools import product
from pathlib import Path

from loguru import logger as logging

from nlnas import (
    TorchvisionClassifier,
    TorchvisionDataset,
    train_and_analyse_all,
)
from nlnas import transforms
from nlnas.utils import targets
from nlnas.transforms import EnsuresRGB
import torchvision


def main():
    model_names = [
        "vit_b_16",
    ]
    submodule_names = [
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
        "model.1",
    ]
    dataset_names = [
        "mnist",
        # "kmnist",
        # "fashionmnist",
        "cifar10",
        # "cifar100",
    ]
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize([224, 224], antialias=True),
            EnsuresRGB(),
        ]
    )
    for m, d in product(model_names, dataset_names):
        output_dir = Path("export-out") / m / d
        ds = TorchvisionDataset(d, transform=transform)
        ds.setup("fit")
        n_classes = len(targets(ds.val_dataloader()))
        image_shape = list(next(iter(ds.val_dataloader()))[0].shape)[1:]
        model = TorchvisionClassifier(
            model_name=m,
            n_classes=n_classes,
            add_final_fc=True,
            input_shape=image_shape,
        )
        train_and_analyse_all(
            model=model,
            submodule_names=submodule_names,
            dataset=ds,
            output_dir=output_dir,
            model_name=m,
            n_samples=500,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except:
        logging.exception(":sad trombone:")
