from itertools import product
from pathlib import Path

import pytorch_lightning as pl
import torchvision.transforms as tvtr
from loguru import logger as logging

from nlnas.classifier import TorchvisionClassifier
from nlnas.logging import setup_logging
from nlnas.nlnas import train_and_analyse_all
from nlnas.training import best_checkpoint_path, train_model_guarded
from nlnas.transforms import *
from nlnas.tv_dataset import DEFAULT_DATALOADER_KWARGS, TorchvisionDataset
from nlnas.utils import best_device


def main():
    pl.seed_everything(0)
    model_names = ["vgg16"]
    analysis_submodules = [
        # "model.0.features.0",
        # "model.0.features.2",
        # "model.0.features.5",
        # "model.0.features.7",
        # "model.0.features.10",
        # "model.0.features.12",
        "model.0.features.14",
        # "model.0.features.17",
        # "model.0.features.19",
        # "model.0.features.21",
        "model.0.features.24",
        # "model.0.features.26",
        # "model.0.features.28",
        "model.0.classifier.0",
        # "model.0.classifier.3",
        "model.0.classifier.6",
    ]
    datasets = {
        "mnist": tvtr.Compose(
            [
                tvtr.ToTensor(),
                EnsureRGB(),
                dataset_normalization("mnist"),
                tvtr.Resize([64, 64], antialias=True),
            ]
        ),
        "fashionmnist": tvtr.Compose(
            [
                tvtr.ToTensor(),
                EnsureRGB(),
                dataset_normalization("fashionmnist"),
                tvtr.Resize([64, 64], antialias=True),
            ]
        ),
        "cifar10": tvtr.Compose(
            [
                tvtr.RandomCrop(32, padding=4),
                tvtr.RandomHorizontalFlip(),
                tvtr.ToTensor(),
                dataset_normalization("cifar10"),
                tvtr.Resize([64, 64], antialias=True),
            ]
        ),
        "cifar100": tvtr.Compose(
            [
                tvtr.RandomCrop(32, padding=4),
                tvtr.RandomHorizontalFlip(),
                tvtr.ToTensor(),
                dataset_normalization("cifar10"),
                tvtr.Resize([64, 64], antialias=True),
            ]
        ),
        "stl10": tvtr.Compose(
            [
                tvtr.RandomHorizontalFlip(),
                tvtr.ToTensor(),
                dataset_normalization("stl10"),
            ]
        ),
        "pcam": tvtr.Compose(
            [
                tvtr.ToTensor(),
                dataset_normalization("pcam"),
            ]
        ),
        "flowers102": tvtr.Compose(
            [
                tvtr.Resize([128, 128], antialias=True),
                tvtr.ToTensor(),
                dataset_normalization("flowers102"),
            ]
        ),
    }
    cor_submodules = [
        "model.0.classifier.0",
        "model.0.classifier.3",
    ]
    weight_exponents = [0, 1, 3, 5, 10]
    batch_sizes = [2048]
    ks = [5, 25, 50]
    for m, (d, t), we, bs, k in product(
        model_names, datasets.items(), weight_exponents, batch_sizes, ks
    ):
        try:
            bcp, _ = best_checkpoint_path(
                f"out/{m}/{d}/model/tb_logs/{m}/version_0/checkpoints/",
                f"out/{m}/{d}/model/csv_logs/{m}/version_0/metrics.csv",
            )
            exp_name = f"{m}_finetune_l{k}_b{bs}_1e-{we}"
            output_dir = Path("out") / exp_name / d
            dataloader_kwargs = DEFAULT_DATALOADER_KWARGS.copy()
            dataloader_kwargs["batch_size"] = bs
            datamodule = TorchvisionDataset(
                d,
                transform=t,
                dataloader_kwargs=dataloader_kwargs,
            )
            model = TorchvisionClassifier.load_from_checkpoint(str(bcp))
            model = model.to(best_device())
            model.cor_type = "louvain"
            model.cor_weight = 10 ** (-we)
            model.cor_submodules = cor_submodules
            model.cor_kwargs = {"k": k}
            train_model_guarded(
                model,
                datamodule,
                output_dir / "model",
                name=exp_name,
                max_epochs=512,
            )
            # train_and_analyse_all(
            #     model=model,
            #     submodule_names=analysis_submodules,
            #     dataset=datamodule,
            #     output_dir=output_dir,
            #     model_name=exp_name,
            # )
        except (KeyboardInterrupt, SystemExit):
            return
        except:
            logging.exception(":sad trombone:")


if __name__ == "__main__":
    setup_logging()
    main()
