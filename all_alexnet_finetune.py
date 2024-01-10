from itertools import product
from pathlib import Path

import pytorch_lightning as pl
import torchvision.transforms as tvtr
from loguru import logger as logging

from nlnas import TorchvisionClassifier, TorchvisionDataset
from nlnas.logging import setup_logging
from nlnas.training import best_checkpoint_path, train_model_guarded


def main():
    pl.seed_everything(0)
    analysis_submodules = [
        "model.0.features.0",
        # "model.0.features.3",
        "model.0.features.6",
        # "model.0.features.8",
        "model.0.features.10",
        # "model.0.features",
        "model.0.classifier.1",
        "model.0.classifier.4",
        "model.0.classifier.6",
        # "model.0.classifier",
        # "model"
    ]
    sep_submodules = [
        "model.0.classifier.1",
        "model.0.classifier.4",
        "model.0.classifier.6",
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
            # EnsuresRGB(),
        ]
    )
    weight_exponents = [1, 2, 3, 4, 5]
    batch_sizes = [1024, 2048, 4096]
    bcp, _ = best_checkpoint_path(
        "out/alexnet/cifar10/model/tb_logs/alexnet/version_0/checkpoints/",
        "out/alexnet/cifar10/model/csv_logs/alexnet/version_0/metrics.csv",
    )
    for we, bs in product(weight_exponents, batch_sizes):
        exp_name = f"alexnet_finetune_l5_b{bs}_1e-{we}"
        output_dir = Path("out") / exp_name / "cifar10"
        datamodule = TorchvisionDataset(
            "cifar10",
            transform=transform,
            dataloader_kwargs={
                "drop_last": True,
                "batch_size": bs,
                "pin_memory": True,
                "num_workers": 4,
                "persistent_workers": True,
            },
        )
        model = TorchvisionClassifier.load_from_checkpoint(str(bcp))
        model.sep_score = "louvain"
        model.sep_weight = 10 ** (-we)
        model.sep_submodules = sep_submodules
        train_model_guarded(
            model,
            datamodule,
            output_dir / "model",
            name=exp_name,
            max_epochs=512,
        )


if __name__ == "__main__":
    setup_logging()
    try:
        main()
    except KeyboardInterrupt:
        pass
    except:
        logging.exception(":sad trombone:")
