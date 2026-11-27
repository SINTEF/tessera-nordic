"""Environment-driven locations and run directories.

All data and output locations come from environment variables (see ``.env.example``):

- ``TN_DATA_DIR``: derived data (per-tile statistics cache, boundary files).
- ``TN_OUTPUT_DIR``: run outputs, ``${TN_OUTPUT_DIR}/<run_name>/<timestamp>/``.
- ``TESSERA_DATA_DIR``: raw TESSERA downloads or the Zarr chunk cache.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

ENV_DATA_DIR = "TN_DATA_DIR"
ENV_OUTPUT_DIR = "TN_OUTPUT_DIR"
ENV_TESSERA_DIR = "TESSERA_DATA_DIR"

_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"


class MissingEnvironmentError(RuntimeError):
    """Raised when a required location environment variable is not set."""


def env_dir(name: str, create: bool = True) -> Path:
    """Return the directory named by environment variable ``name``.

    Args:
        name: Environment variable name.
        create: Create the directory if it does not exist.

    Raises:
        MissingEnvironmentError: if the variable is unset or empty.
    """
    value = os.environ.get(name, "").strip()
    if not value:
        raise MissingEnvironmentError(
            f"Environment variable {name} is not set. Copy .env.example to .env "
            "and set it to a directory outside the repository."
        )
    path = Path(value).expanduser().resolve()
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir() -> Path:
    """Directory for derived data (``TN_DATA_DIR``)."""
    return env_dir(ENV_DATA_DIR)


def output_dir() -> Path:
    """Directory for run outputs (``TN_OUTPUT_DIR``)."""
    return env_dir(ENV_OUTPUT_DIR)


def tessera_dir() -> Path:
    """Directory for raw TESSERA data (``TESSERA_DATA_DIR``)."""
    return env_dir(ENV_TESSERA_DIR)


def new_run_dir(run_name: str, now: datetime | None = None) -> Path:
    """Create ``${TN_OUTPUT_DIR}/<run_name>/<timestamp>/`` and return it.

    Args:
        run_name: Run name from the config.
        now: Override the timestamp (for tests).
    """
    stamp = (now or datetime.now(UTC)).strftime(_TIMESTAMP_FORMAT)
    path = output_dir() / run_name / stamp
    suffix = 1
    while path.exists():  # two runs started in the same second
        path = output_dir() / run_name / f"{stamp}-{suffix}"
        suffix += 1
    path.mkdir(parents=True)
    return path


def latest_run_dir(run_name: str) -> Path:
    """Return the most recent run directory for ``run_name``.

    Raises:
        FileNotFoundError: if no run exists yet.
    """
    base = output_dir() / run_name
    runs = sorted(p for p in base.glob("*") if p.is_dir()) if base.exists() else []
    if not runs:
        raise FileNotFoundError(
            f"No runs found under {base}. Run `tessera-nordic aggregate` (or `run`) first."
        )
    return runs[-1]
