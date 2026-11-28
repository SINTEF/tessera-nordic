"""Reproducibility records written into every run directory."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from tessera_nordic.config import RunConfig, dump_config

PACKAGES = [
    "tessera-nordic",
    "geotessera",
    "numpy",
    "pandas",
    "pyarrow",
    "geopandas",
    "shapely",
    "pyproj",
    "rasterio",
    "scikit-learn",
    "zarr",
    "xarray",
    "pydantic",
    "typer",
]


def git_info(repo_dir: Path | None = None) -> dict[str, Any]:
    """Current commit hash and dirty flag; falls back to ``$TN_GIT_COMMIT`` (Docker)."""
    cwd = repo_dir or Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": os.environ.get("TN_GIT_COMMIT", "unknown"), "dirty": None}


def package_versions() -> dict[str, str]:
    """Installed versions of the packages that affect results."""
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def set_seeds(seed: int) -> None:
    """Seed Python's and NumPy's global generators.

    Estimators additionally receive ``random_state`` explicitly.
    """
    random.seed(seed)
    np.random.seed(seed)


def write_run_record(
    run_dir: Path,
    config: RunConfig,
    config_path: Path,
    command: str,
    sources: dict[str, Any] | None = None,
) -> None:
    """Write ``config.resolved.yaml`` and ``provenance.json`` into ``run_dir``.

    If ``provenance.json`` exists, the new command is appended to its history.

    Args:
        run_dir: Run directory.
        config: Resolved configuration.
        config_path: Path of the YAML file it came from.
        command: CLI stage being run.
        sources: Provenance of feature sources (dataset versions etc.).
    """
    dump_config(config, run_dir / "config.resolved.yaml")
    path = run_dir / "provenance.json"
    record: dict[str, Any]
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
    else:
        record = {
            "run_name": config.run_name,
            "run_dir": str(run_dir),
            "created_utc": datetime.now(UTC).isoformat(),
            "config_file": str(config_path),
            "year": config.year,
            "seed": config.seed,
            "git": git_info(),
            "python": sys.version,
            "platform": platform.platform(),
            "packages": package_versions(),
            "sources": {},
            "commands": [],
        }
    if sources:
        record["sources"].update(sources)
        tessera = sources.get("tessera")
        if tessera:
            record["tessera_embedding_version"] = {
                k: tessera.get(k)
                for k in (
                    "dataset_version",
                    "dataset_variant",
                    "model_version",
                    "store_build_version",
                )
                if k in tessera
            }
    record["commands"].append({"command": command, "utc": datetime.now(UTC).isoformat()})
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
