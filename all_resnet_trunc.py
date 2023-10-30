from itertools import product
from pathlib import Path

import pytorch_lightning as pl
import torchvision.transforms as tvtr
from loguru import logger as logging

from nlnas import (
    TorchvisionClassifier,
    TorchvisionDataset,
    train_and_analyse_all,
)
from nlnas.classifier import TorchvisionClassifier, TruncatedClassifier
from nlnas.logging import setup_logging
from nlnas.training import (
    best_checkpoint_path,
    train_model,
    train_model_guarded,
)
from nlnas.utils import dataset_n_targets


def main():
    pl.seed_everything(0)

    ckpt, _ = best_checkpoint_path(
        "out/resnet18/cifar10/model/tb_logs/resnet18/version_0/checkpoints",
        "out/resnet18/cifar10/model/csv_logs/resnet18/version_0/metrics.csv",
    )

    # MNIST
    # model_name, dataset_name = "resnet18_layer3", "mnist"
    # resnet18 = TorchvisionClassifier.load_from_checkpoint(
    #     "out/resnet18/mnist/model/tb_logs/resnet18/version_0/checkpoints/epoch=9-step=940.ckpt"
    # )
    # model = TruncatedClassifier(
    #     model=resnet18,
    #     truncate_after="model.0.layer3",
    #     n_classes=resnet18.hparams["n_classes"],
    #     input_shape=resnet18.hparams["input_shape"],
    # )

    # FASHIONMNIST
    # model_name, dataset_name = "resnet18_layer4", "fashionmnist"
    # resnet18 = TorchvisionClassifier.load_from_checkpoint(
    #     "out/resnet18/fashionmnist/model/tb_logs/resnet18/version_0/checkpoints/epoch=6-step=658.ckpt"
    # )
    # model = TruncatedClassifier(
    #     model=resnet18,
    #     truncate_after="model.0.layer4",
    #     n_classes=resnet18.hparams["n_classes"],
    #     input_shape=resnet18.hparams["input_shape"],
    # )

    # CIFAR10
    model_name, dataset_name = "resnet18_layer4", "cifar10"
    ckpt, _ = best_checkpoint_path(
        "out/resnet18/cifar10/model/tb_logs/resnet18/version_0/checkpoints",
        "out/resnet18/cifar10/model/csv_logs/resnet18/version_0/metrics.csv",
    )
    truncated_model = TruncatedClassifier(
        model=ckpt,
        truncate_after="model.0.layer4",
    )
    submodule_names = [
        "model.model.0.layer1",
        "model.model.0.layer2",
        "model.model.0.layer3",
        "model.model.0.layer4",
        "fc",
    ]

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
    ds = TorchvisionDataset(dataset_name, transform=transform)

    output_dir = Path("out") / model_name / dataset_name
    # train_model(
    #     truncated_model,
    #     ds,
    #     output_dir / "model",
    #     name=model_name,
    #     max_epochs=512,
    #     # strategy="ddp_find_unused_parameters_true",  # if unfrozen
    #     reload=False,
    # )
    train_and_analyse_all(
        model=truncated_model,
        submodule_names=submodule_names,
        dataset=ds,
        output_dir=output_dir,
        model_name=model_name,
    )


if __name__ == "__main__":
    setup_logging()
    try:
        main()
    except KeyboardInterrupt:
        pass
    except:
        logging.exception(":sad trombone:")
