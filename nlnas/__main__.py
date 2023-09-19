"""CLI module"""


import os
import sys
from typing import Literal

import click
from loguru import logger as logging


def _setup_logging(
    logging_level: Literal[
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
    ]
) -> None:
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
def main(logging_level: str):
    """Entrypoint."""
    _setup_logging(logging_level)


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    main()
