from itertools import product
from pathlib import Path

import pytorch_lightning as pl
from _parameters import BATCH_SIZES, DATASETS, KS, WEIGHT_EXPONENTS
from loguru import logger as logging

from nlnas import (
    DEFAULT_DATALOADER_KWARGS,
    TorchvisionClassifier,
    TorchvisionDataset,
    best_checkpoint_path,
    best_device,
    train_and_analyse_all,
    train_model_guarded,
)
from nlnas.logging import setup_logging


def main():
    pl.seed_everything(0)
    model_names = ["vgg16"]
    analysis_submodules = [
        # "model.features.0",
        # "model.features.2",
        # "model.features.5",
        # "model.features.7",
        # "model.features.10",
        # "model.features.12",
        "model.features.14",
        # "model.features.17",
        # "model.features.19",
        # "model.features.21",
        "model.features.24",
        # "model.features.26",
        # "model.features.28",
        "model.classifier.0",
        # "model.classifier.3",
        "model.classifier.6",
    ]
    cor_submodules = [
        "model.classifier.0",
        "model.classifier.3",
    ]
    for m, (d, t), we, bs, k in product(
        model_names, DATASETS.items(), WEIGHT_EXPONENTS, BATCH_SIZES, KS
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
