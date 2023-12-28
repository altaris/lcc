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
from nlnas.utils import dl_targets


def main():
    pl.seed_everything(0)

    dataset_name = "cifar10"
    base_model_name = "resnet18_lv_l1l2l3l4fc1_w1_b256"
    trunc_model_name = "resnet18_lv_l1l2l3l4fc1_w1_b256_l3_uf"
    submodule_names = [
        "model.model.0.layer1",
        "model.model.0.layer2",
        "model.model.0.layer3",
        "fc",
    ]

    ckpt, _ = best_checkpoint_path(
        f"out/{base_model_name}/{dataset_name}/model/tb_logs/{base_model_name}/version_0/checkpoints",
        f"out/{base_model_name}/{dataset_name}/model/csv_logs/{base_model_name}/version_0/metrics.csv",
    )
    truncated_model = TruncatedClassifier(
        model=ckpt,
        truncate_after="model.0.layer3",
        sep_submodules=submodule_names,
        sep_score="lv",
        sep_weight=1,
        # freeze_base_model=False,
    )

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

    output_dir = Path("out") / trunc_model_name / dataset_name
    train_and_analyse_all(
        model=truncated_model,
        submodule_names=submodule_names,
        dataset=ds,
        output_dir=output_dir,
        model_name=trunc_model_name,
        # strategy="ddp_find_unused_parameters_true",
    )


if __name__ == "__main__":
    setup_logging()
    try:
        main()
    except KeyboardInterrupt:
        pass
    except:
        logging.exception(":sad trombone:")
