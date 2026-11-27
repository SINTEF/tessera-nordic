"""Logging setup: console plus a ``run.log`` file in the run directory."""

from __future__ import annotations

import logging
import logging.handlers
import multiprocessing as mp
from pathlib import Path
from typing import Any

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(processName)s %(name)s: %(message)s"
LOGGER_NAME = "tessera_nordic"


def setup_logging(run_dir: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure the root logger to write to the console and, optionally, a run log.

    Calling it again replaces previously installed handlers, so it is safe to call
    once per CLI command.

    Args:
        run_dir: If given, also append to ``run_dir / "run.log"``.
        level: Log level for our own loggers.

    Returns:
        The package logger.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter(LOG_FORMAT)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)
    if run_dir is not None:
        file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    root.setLevel(logging.WARNING)  # third-party libraries stay quiet by default
    logging.getLogger(LOGGER_NAME).setLevel(level)
    logging.getLogger("geotessera").setLevel(logging.WARNING)
    logging.captureWarnings(True)
    return logging.getLogger(LOGGER_NAME)


def start_queue_listener() -> tuple[Any, logging.handlers.QueueListener]:
    """Start a listener that forwards worker-process log records to the root handlers.

    Returns:
        The queue to hand to worker processes and the running listener (stop it when done).
    """
    queue: Any = mp.get_context("spawn").Manager().Queue(-1)
    listener = logging.handlers.QueueListener(
        queue, *logging.getLogger().handlers, respect_handler_level=True
    )
    listener.start()
    return queue, listener


def configure_worker_logging(queue: Any, level: int = logging.INFO) -> None:
    """Route a worker process's log records into ``queue``.

    Args:
        queue: Queue returned by :func:`start_queue_listener`.
        level: Level for the package logger in the worker.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.addHandler(logging.handlers.QueueHandler(queue))
    root.setLevel(logging.WARNING)
    logging.getLogger(LOGGER_NAME).setLevel(level)
