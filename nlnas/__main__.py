"""CLI module"""


import os
from pathlib import Path
import sys

import click
from loguru import logger as logging


def _setup_logging(logging_level: str) -> None:
    """
    Sets logging format and level. The format is

        %(asctime)s [%(levelname)-8s] %(message)s

    e.g.

        2022-02-01 10:41:43,797 [INFO    ] Hello world
        2022-02-01 10:42:12,488 [CRITICAL] We're out of beans!

    Args:
        logging_level (str): Either 'critical', 'debug', 'error', 'info', or
            'warning', case insensitive. If invalid, defaults to 'info'.
    """
    logging.remove()
    logging.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
            + "[<level>{level: <8}</level>] "
            + "<level>{message}</level>"
        ),
        level=logging_level.upper(),
        enqueue=True,
        colorize=True,
    )


@click.command()
@click.argument("model_name", type=str)
@click.argument("submodule_names", type=str)
@click.argument("dataset_name", type=str)
@click.argument(
    "output_dir", type=click.Path(exists=True, file_okay=False, writable=True)
)
@click.option("--n-samples", default=10000, type=int)
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
    n_samples: int,
    logging_level: str,
):
    """Entrypoint"""

    from .nlnas import training_suite

    _setup_logging(logging_level)

    training_suite(
        model_name=model_name,
        submodule_names=submodule_names,
        dataset_name=dataset_name,
        output_dir=output_dir,
        n_samples=n_samples,
    )


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    main()
