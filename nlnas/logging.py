"""Logging utilities (duh)"""

import sys

from loguru import logger as logging
from pytorch_lightning.utilities.rank_zero import rank_zero_only


@rank_zero_only
def r0_debug(message: str, *args, **kwargs) -> None:
    """Logs a message with severity `DEBUG` but only in the rank 0 process"""
    logging.debug(message, *args, **kwargs)


@rank_zero_only
def r0_critical(message: str, *args, **kwargs) -> None:
    """Logs a message with severity `CRITICAL` but only in the rank 0 process"""
    logging.critical(message, *args, **kwargs)


@rank_zero_only
def r0_error(message: str, *args, **kwargs) -> None:
    """Logs a message with severity `ERROR` but only in the rank 0 process"""
    logging.error(message, *args, **kwargs)


@rank_zero_only
def r0_info(message: str, *args, **kwargs) -> None:
    """Logs a message with severity `INFO` but only in the rank 0 process"""
    logging.info(message, *args, **kwargs)


@rank_zero_only
def r0_success(message: str, *args, **kwargs) -> None:
    """Logs a message with severity `SUCCESS` but only in the rank 0 process"""
    logging.success(message, *args, **kwargs)


@rank_zero_only
def r0_trace(message: str, *args, **kwargs) -> None:
    """Logs a message with severity `TRACE` but only in the rank 0 process"""
    logging.trace(message, *args, **kwargs)


@rank_zero_only
def r0_warning(message: str, *args, **kwargs) -> None:
    """Logs a message with severity `WARNING` but only in the rank 0 process"""
    logging.warning(message, *args, **kwargs)


def setup_logging(logging_level: str = "info") -> None:
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
