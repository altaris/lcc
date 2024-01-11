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
from nlnas.tv_dataset import DEFAULT_DATALOADER_KWARGS, TorchvisionDataset


def main():
    pl.seed_everything(0)
    analysis_submodules = [
        # "model.0.maxpool",
        "model.0.layer1",
        "model.0.layer2",
        "model.0.layer3",
        "model.0.layer4",
        "model.0.fc",
    ]
    sep_submodules = [
        # "model.0.layer3",
        "model.0.layer4",
        "model.0.fc",
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
    bcp, _ = best_checkpoint_path(
        "out/resnet18/cifar10/model/tb_logs/resnet18/version_0/checkpoints/",
        "out/resnet18/cifar10/model/csv_logs/resnet18/version_0/metrics.csv",
    )
    for we, bs in product(weight_exponents, batch_sizes):
        try:
            exp_name = f"resnet_finetune_l5_b{bs}_1e-{we}"
            output_dir = Path("out") / exp_name / "cifar10"
            dataloader_kwargs = DEFAULT_DATALOADER_KWARGS.copy()
            dataloader_kwargs["batch_size"] = 2048
            datamodule = TorchvisionDataset(
                "cifar10",
                transform=transform,
                dataloader_kwargs=dataloader_kwargs,
            )
            model = TorchvisionClassifier.load_from_checkpoint(str(bcp))
            model.sep_score = "louvain"
            model.sep_weight = 10 ** (-we)
            model.sep_submodules = sep_submodules
            # train_model_guarded(
            #     model,
            #     datamodule,
            #     output_dir / "model",
            #     name=exp_name,
            #     max_epochs=512,
            # )
            train_and_analyse_all(
                model=model,
                submodule_names=analysis_submodules,
                dataset=datamodule,
                output_dir=output_dir,
                model_name=exp_name,
            )
        except KeyboardInterrupt:
            pass
        except:
            logging.exception(":sad trombone:")


if __name__ == "__main__":
    setup_logging()
    main()
