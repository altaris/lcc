from itertools import product
from pathlib import Path

from loguru import logger as logging

from nlnas import (
    TorchvisionClassifier,
    TorchvisionDataset,
    train_and_analyse_all,
)
from nlnas.utils import targets


def main():
    model_names = [
        "resnet18",
        # "resnet34",
        # "resnet50",
        # "resnet101",
        # "resnet152",
    ]
    submodule_names = [
        "model.0.maxpool",
        "model.0.layer1",
        "model.0.layer2",
        "model.0.layer3",
        "model.0.layer4",
        "model.0.fc",
        "model.1",
    ]
    dataset_names = [
        "mnist",
        "kmnist",
        "fashionmnist",
        "cifar10",
        "cifar100",
    ]
    for m, d in product(model_names, dataset_names):
        output_dir = Path("out") / m / d
        ds = TorchvisionDataset(d)
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
