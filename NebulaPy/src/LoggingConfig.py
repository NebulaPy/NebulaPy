"""Central logging configuration for NebulaPy applications."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Union

from rich.console import Console
from rich.logging import RichHandler

from NebulaPy import version


LOGGER_NAME = "NebulaPy"
DEFAULT_MAX_BYTES = 10_000_000
DEFAULT_BACKUP_COUNT = 2
_DEFAULT_EXCEPTHOOK = sys.excepthook
CONSOLE = Console(stderr=True)


class NebulaError(RuntimeError):
    """Base exception for recoverable NebulaPy runtime failures."""


def _log_unhandled_exception(exc_type, exc_value, traceback):
    """Record uncaught application exceptions through NebulaPy logging."""
    if issubclass(exc_type, KeyboardInterrupt):
        _DEFAULT_EXCEPTHOOK(exc_type, exc_value, traceback)
        return
    get_logger().critical(
        "Unhandled exception",
        exc_info=(exc_type, exc_value, traceback),
    )


class _ScriptContextFilter(logging.Filter):
    """Attach the invoking script name to records written to the log file."""

    def __init__(self, script_name: str):
        super().__init__()
        self.script_name = script_name

    def filter(self, record: logging.LogRecord) -> bool:
        record.script_name = self.script_name
        return True


class _ConsoleRecordFilter(logging.Filter):
    """Exclude records intended only for the detailed log file."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not getattr(record, "nebulapy_file_only", False)


def _coerce_level(level: Union[int, str]) -> int:
    if isinstance(level, int):
        return level

    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unknown logging level: {level}")
    return numeric_level


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger inside the shared NebulaPy namespace."""
    if not name:
        return logging.getLogger(LOGGER_NAME)
    if name == LOGGER_NAME or name.startswith(f"{LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def logging_is_configured() -> bool:
    """Return whether the current process has configured NebulaPy logging."""
    return bool(getattr(get_logger(), "_nebulapy_configured", False))


def configure_logging(
    level: Union[int, str] = "INFO",
    *,
    file_level: Union[int, str] = "DEBUG",
    log_file: str | Path | None = None,
    log_to_file: bool = True,
    console: bool = True,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
) -> Path | None:
    """Configure shared console and rotating-file logging.

    By default, logs are written to ``Path.cwd() / "NebulaPy.log"`` so their
    location follows the directory from which the application is launched.
    A relative ``log_file`` name is resolved from that working directory.
    Pass ``log_to_file=False`` to use terminal logging without creating a log
    file. Repeated calls replace only handlers installed by this function.
    """
    console_level = _coerce_level(level)
    disk_level = _coerce_level(file_level)

    destination = None
    if log_to_file:
        destination = (
            Path(log_file).expanduser()
            if log_file is not None
            else Path.cwd() / "NebulaPy.log"
        )
        destination = destination.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)

    logger = get_logger()
    active_levels = []
    if console:
        active_levels.append(console_level)
    if log_to_file:
        active_levels.append(disk_level)
    logger.setLevel(min(active_levels) if active_levels else logging.CRITICAL + 1)
    logger.propagate = False

    for handler in list(logger.handlers):
        if getattr(handler, "_nebulapy_handler", False):
            logger.removeHandler(handler)
            handler.close()

    script_name = Path(sys.argv[0]).name or "interactive"

    if console:
        console_handler = RichHandler(
            console=CONSOLE,
            show_time=True,
            omit_repeated_times=False,
            show_level=True,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            log_time_format="[%H:%M:%S]",
            markup=False,
        )
        console_handler._nebulapy_handler = True
        console_handler.setLevel(console_level)
        console_handler.addFilter(_ConsoleRecordFilter())
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(console_handler)

    if destination is not None:
        file_handler = RotatingFileHandler(
            destination,
            mode="a",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
            delay=True,
        )
        file_handler._nebulapy_handler = True
        file_handler.setLevel(disk_level)
        file_handler.addFilter(_ScriptContextFilter(script_name))
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(script_name)s | "
            "%(name)s.%(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(file_handler)
    logger._nebulapy_configured = True
    logger._nebulapy_log_file = destination
    sys.excepthook = _log_unhandled_exception

    if destination is not None:
        logger.info(
            "NebulaPy %s started",
            version.__version__,
            extra={"nebulapy_file_only": True},
        )
    logger.debug("Working directory: %s", Path.cwd())
    if destination is not None:
        logger.debug("Log file: %s", destination)

    return destination
