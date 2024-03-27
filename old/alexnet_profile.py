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
    sep_submodules = [
        "model.classifier.1",
        "model.classifier.4",
        "model.classifier.6",
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
    bcp, _ = best_checkpoint_path(
        "out/alexnet/cifar10/model/tb_logs/alexnet/version_0/checkpoints/",
        "out/alexnet/cifar10/model/csv_logs/alexnet/version_0/metrics.csv",
    )
    bs, we = 4096, 3
    exp_name = f"alexnet_finetune_l5_b{bs}_1e-{we}"
    output_dir = Path("prof") / exp_name / "cifar10"
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
        max_epochs=10,
    )


if __name__ == "__main__":
    setup_logging()
    main()
