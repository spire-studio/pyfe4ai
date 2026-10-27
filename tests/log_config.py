"""Logging of the test suite.

The library only adds a ``NullHandler`` to the ``pyfe4ai`` logger; the tests
that want the scheme logs call :func:`init_logging_config`: DEBUG to the
console (captured by pytest) and INFO / ERROR to rotating files in
``$PYFE4AI_TEST_LOG_DIR``, by default ``pyfe4ai-tests/logs`` in the system's
temporary directory (not the working directory).
"""

from __future__ import annotations

import os
import tempfile
from logging.config import dictConfig

#: Environment variable naming the directory of the log files.
LOG_DIR_ENV = "PYFE4AI_TEST_LOG_DIR"


def log_dir() -> str:
    """The directory of the test log files."""
    return os.environ.get(LOG_DIR_ENV) or os.path.join(tempfile.gettempdir(), "pyfe4ai-tests", "logs")


def logging_config(directory: str) -> dict:
    """The ``dictConfig`` of the tests, with the log files in *directory*."""

    def file_handler(level: str, name: str) -> dict:
        return {
            "class": "logging.handlers.RotatingFileHandler",
            "level": level,
            "formatter": "formatter1",
            "filename": os.path.join(directory, name),
            "maxBytes": 10485760,  # 10 MB
            "backupCount": 10,
            "encoding": "utf8",
        }

    handlers = ["console", "info_file_handler", "error_file_handler"]
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "formatter1": {"format": "%(asctime)s | %(levelname)-8s | %(name)-45s | %(message)s"},
            "formatter2": {"format": "%(asctime)s | %(levelname)-8s | %(name)-25s |%(funcName)-30s | %(message)s"},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "formatter": "formatter2",
                "stream": "ext://sys.stdout",
            },
            "info_file_handler": file_handler("INFO", "info.log"),
            "error_file_handler": file_handler("ERROR", "error.log"),
        },
        "loggers": {"default": {"level": "DEBUG", "handlers": handlers, "propagate": False}},
        "root": {"level": "DEBUG", "handlers": handlers},
    }


def init_logging_config() -> None:
    """Configure logging for the tests (see the module docstring)."""
    directory = log_dir()
    os.makedirs(directory, exist_ok=True)
    dictConfig(logging_config(directory))
