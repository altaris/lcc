from itertools import product
from pathlib import Path

import pytorch_lightning as pl
from loguru import logger as logging
from torch import Tensor

from nlnas import (
    TorchvisionClassifier,
    TorchvisionDataset,
    train_and_analyse_all,
)
from nlnas.utils import targets


def extract_logits(_module, _inputs, outputs) -> Tensor | None:
    """Googlenet outputs a named tuple instead of a tensor"""
    return outputs.logits if not isinstance(outputs, Tensor) else None


def main():
    pl.seed_everything(0)
    model_names = [
        "googlenet",
    ]
    submodule_names = [
        "model.0.maxpool1",
        # "model.0.maxpool2",
        "model.0.inception3a",
        "model.0.maxpool3",
        # "model.0.inception4a",
        # "model.0.inception4b",
        # "model.0.inception4c",
        "model.0.inception4d",
        "model.0.maxpool4",
        # "model.0.inception5a",
        "model.0.inception5b",
        "model.0.fc",
        "model.1",
    ]
    dataset_names = [
        "mnist",
        # "kmnist",
        "fashionmnist",
        "cifar10",
        # "cifar100",
    ]
    for m, d in product(model_names, dataset_names):
        output_dir = Path("out") / m / d
        ds = TorchvisionDataset(d)
        ds.setup("fit")
        n_classes = len(targets(ds.val_dataloader()))
        image_shape = list(next(iter(ds.val_dataloader()))[0].shape)[1:]
        model = TorchvisionClassifier(
            model_name=m,
            n_classes=n_classes,
            add_final_fc=True,
            input_shape=image_shape,
        )
        model.model[0].register_forward_hook(extract_logits)
        train_and_analyse_all(
            model=model,
            submodule_names=submodule_names,
            dataset=ds,
            output_dir=output_dir,
            model_name=m,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except:
        logging.exception(":sad trombone:")
