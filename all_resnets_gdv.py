from itertools import product
from pathlib import Path

import pytorch_lightning as pl
import torchvision
from loguru import logger as logging

from nlnas import (
    TorchvisionClassifier,
    TorchvisionDataset,
    VHTorchvisionClassifier,
    train_and_analyse_all,
)
from nlnas.training import train_model, train_model_guarded
from nlnas.utils import targets


def main():
    pl.seed_everything(0)
    model_names = [
        "resnet18",
    ]
    submodule_names = [
        # "model.0.layer1.0"
        # "model.0.layer1.1"
        # "model.0.layer1",
        # "model.0.layer2.0"
        # "model.0.layer2.1"
        # "model.0.layer2",
        # "model.0.layer3.0"
        # "model.0.layer3.1"
        # "model.0.layer3",
        # "model.0.layer4.0"
        "model.0.layer4.1"
        # "model.0.layer4",
        # "model.0.fc",
        # "model.1",
    ]
    dataset_names = [
        # "mnist",
        # "kmnist",
        # "fashionmnist",
        "cifar10",
        # "cifar100",
    ]
    for model_name, dataset_name in product(model_names, dataset_names):
        output_dir = Path("out") / (model_name + "_gdv") / dataset_name
        dataset = TorchvisionDataset(dataset_name)
        dataset.setup("fit")
        n_classes = len(targets(dataset.val_dataloader()))
        image_shape = list(next(iter(dataset.val_dataloader()))[0].shape)[1:]
        model = VHTorchvisionClassifier(
            model_name=model_name,
            n_classes=n_classes,
            vh_submodules=submodule_names,
            add_final_fc=True,
            input_shape=image_shape,
            horizontal_lr=5e-3,
        )
        # train_model_guarded(
        train_model(
            model,
            dataset,
            output_dir / "model",
            name=model_name,
            max_epochs=512,
            strategy="ddp_find_unused_parameters_true",
        )
        # train_and_analyse_all(
        #     model=model,
        #     submodule_names=submodule_names,
        #     dataset=ds,
        #     output_dir=output_dir,
        #     model_name=m + "_gdv",
        # )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except:
        logging.exception(":sad trombone:")
