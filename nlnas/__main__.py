"""CLI module"""
# pylint: disable=import-outside-toplevel

import os
from pathlib import Path

import click
from loguru import logger as logging

from .logging import setup_logging


@click.command()
@click.argument("model_name", type=str)
@click.argument("submodule_names", type=str)
@click.argument("dataset_name", type=str)
@click.argument(
    "output_dir", type=click.Path(exists=True, file_okay=False, writable=True)
)
@click.option(
    "--logging-level",
    default=os.getenv("LOGGING_LEVEL", "info"),
    help=(
        "Logging level, among 'critical', 'debug', 'error', 'info', and "
        "'warning', case insensitive."
    ),
    type=click.Choice(
        [
            "critical",
            "CRITICAL",
            "debug",
            "DEBUG",
            "error",
            "ERROR",
            "info",
            "INFO",
            "warning",
            "WARNING",
        ],
        case_sensitive=False,
    ),
)
@logging.catch
def main(
    model_name: str,
    submodule_names: str,
    dataset_name: str,
    output_dir: Path,
    logging_level: str,
):
    """Entrypoint"""

    import pytorch_lightning as pl

    from .classifier import TorchvisionClassifier
    from .nlnas import train_and_analyse_all
    from .tv_dataset import TorchvisionDataset
    from .utils import dataset_n_targets

    setup_logging(logging_level)
    pl.seed_everything(0)

    output_dir = Path("out") / model_name / dataset_name
    ds = TorchvisionDataset(dataset_name)
    ds.setup("fit")
    train_and_analyse_all(
        model=TorchvisionClassifier(
            model_name,
            n_classes=len(dataset_n_targets(ds.train_dataloader())),
        ),
        submodule_names=submodule_names.split(","),
        dataset=ds,
        output_dir=output_dir,
        model_name=model_name,
    )


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    main()
