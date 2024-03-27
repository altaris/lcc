from itertools import product
from pathlib import Path

import pytorch_lightning as pl
from _parameters import DATASETS
from loguru import logger as logging
from torch import Tensor

from nlnas import (
    DEFAULT_DATALOADER_KWARGS,
    TorchvisionClassifier,
    TorchvisionDataset,
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
    for m, (d, t) in product(model_names, DATASETS.items()):
        try:
            output_dir = Path("out") / m / d
            dataloader_kwargs = DEFAULT_DATALOADER_KWARGS.copy()
            dataloader_kwargs["batch_size"] = 2048
            datamodule = TorchvisionDataset(
                d,
                transform=t,
                dataloader_kwargs=dataloader_kwargs,
            )
            model = TorchvisionClassifier(
                model_name=m,
                n_classes=datamodule.n_classes,
                input_shape=datamodule.image_shape,
            )
            model = model.to(best_device())
            model.model[0].register_forward_hook(extract_logits)  # type: ignore
            train_model_guarded(
                model,
                datamodule,
                output_dir / "model",
                name=m,
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
            pass
        except:
            logging.exception(":sad trombone:")


if __name__ == "__main__":
    setup_logging()
    main()
