from itertools import product
from pathlib import Path

import torchvision
from loguru import logger as logging

from nlnas import (
    TorchvisionClassifier,
    TorchvisionDataset,
    train_and_analyse_all,
)
from nlnas import transforms
from nlnas.transforms import EnsuresRGB
from nlnas.utils import targets


def main():
    model_names = [
        "alexnet",
    ]
    submodule_names = [
        "model.0.features.2",
        # "model.0.features.5",
        "model.0.features.7",
        # "model.0.features.9",
        "model.0.avgpool",
        "model.0.classifier.2",
        # "model.0.classifier.5",
        "model.0.classifier.6",
        "model.1",
    ]
    dataset_names = [
        # "mnist",
        "kmnist",
        # "fashionmnist",
        "cifar10",
        # "cifar100",
    ]
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize([64, 64], antialias=True),
            EnsuresRGB(),
        ]
    )
    for m, d in product(model_names, dataset_names):
        output_dir = Path("out") / m / d
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
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except:
        logging.exception(":sad trombone:")
