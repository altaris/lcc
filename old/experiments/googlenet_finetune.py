from itertools import product
from pathlib import Path

import pytorch_lightning as pl
from _parameters import BATCH_SIZES, DATASETS, KS, WEIGHT_EXPONENTS
from loguru import logger as logging
from torch import Tensor

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


def extract_logits(_module, _inputs, outputs) -> Tensor | None:
    """Googlenet outputs a named tuple instead of a tensor"""
    return outputs.logits if not isinstance(outputs, Tensor) else None


def main():
    pl.seed_everything(0)
    model_names = ["googlenet"]
    analysis_submodules = [
        "model.conv1",
        "model.conv2",
        "model.conv3",
        "model.inception3a",
        "model.inception4a",
        "model.inception4b",
        "model.inception4c",
        "model.inception4d",
        "model.inception5a",
        "model.inception5b",
        "model.fc",
    ]
    cor_submodules = [
        "model.inception5a",
        "model.inception5b",
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
