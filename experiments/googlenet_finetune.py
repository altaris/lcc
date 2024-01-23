from itertools import product
from pathlib import Path

import pytorch_lightning as pl
import torchvision.transforms as tvtr
from loguru import logger as logging
from torch import Tensor

from nlnas.classifier import TorchvisionClassifier
from nlnas.logging import setup_logging
from nlnas.nlnas import train_and_analyse_all
from nlnas.training import best_checkpoint_path, train_model_guarded
from nlnas.transforms import EnsureRGB, dataset_normalization
from nlnas.tv_dataset import DEFAULT_DATALOADER_KWARGS, TorchvisionDataset
from nlnas.utils import best_device


def extract_logits(_module, _inputs, outputs) -> Tensor | None:
    """Googlenet outputs a named tuple instead of a tensor"""
    return outputs.logits if not isinstance(outputs, Tensor) else None


def main():
    pl.seed_everything(0)
    model_names = ["googlenet"]
    analysis_submodules = [
        "model.0.conv1",
        "model.0.conv2",
        "model.0.conv3",
        "model.0.inception3a",
        "model.0.inception4a",
        "model.0.inception4b",
        "model.0.inception4c",
        "model.0.inception4d",
        "model.0.inception5a",
        "model.0.inception5b",
        "model.0.fc",
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
        "model.0.inception5a",
        "model.0.inception5b",
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
            model.model[0].register_forward_hook(extract_logits)
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
                strategy="ddp_find_unused_parameters_true",
            )
            # train_and_analyse_all(
            #     model=model,
            #     submodule_names=analysis_submodules,
            #     dataset=datamodule,
            #     output_dir=output_dir,
            #     model_name=m,
            #     strategy="ddp_find_unused_parameters_true",
            # )
        except (KeyboardInterrupt, SystemExit):
            return
        except:
            logging.exception(":sad trombone:")


if __name__ == "__main__":
    setup_logging()
    main()
