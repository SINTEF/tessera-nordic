"""``tessera-nordic`` command-line interface: one subcommand per pipeline stage."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer
from dotenv import load_dotenv

from tessera_nordic import paths
from tessera_nordic.config import RunConfig, load_config
from tessera_nordic.logging_utils import setup_logging
from tessera_nordic.provenance import set_seeds, write_run_record

app = typer.Typer(
    help="Cluster TESSERA embeddings over Norway and Sweden on a 1 km grid.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
log = logging.getLogger("tessera_nordic.cli")

ConfigArg = Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Run config YAML.")]
RunDirOpt = Annotated[
    Path | None,
    typer.Option(
        "--run-dir",
        help="Existing run directory to use. Default: a new one for aggregate/run, "
        "the latest one of this run_name for later stages.",
    ),
]
WorkersOpt = Annotated[int | None, typer.Option("--workers", "-w", min=1, help="Override workers.")]

# Rough size of one 0.1-degree tile at 10 m (int8 embeddings + scales) at ~63 N.
_BYTES_PER_TILE = 77e6  # decoded int8 embeddings + scales of one tile at ~63 N
# Measured network transfer per tile from the v1.1 dclimate Zarr store (2026-11-28):
# 240-320 MB, i.e. ~3.4x the decoded size (sub-shard reads fetch more than the window).
_NETWORK_BYTES_PER_TILE = 260e6


def _prepare(config_path: Path, run_dir: Path | None, new: bool) -> tuple[RunConfig, Path]:
    load_dotenv(override=False)
    config = load_config(config_path)
    if run_dir is None:
        run_dir = (
            paths.new_run_dir(config.run_name) if new else paths.latest_run_dir(config.run_name)
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir)
    set_seeds(config.seed)
    log.info("Run directory: %s", run_dir)
    return config, run_dir


def _guard(command: str, fn: Callable[[], Any]) -> Any:
    start = time.monotonic()
    try:
        result = fn()
    except Exception as exc:
        log.exception("%s failed: %s", command, exc)
        raise typer.Exit(code=1) from exc
    log.info("%s finished in %.1f s", command, time.monotonic() - start)
    return result


def _source_provenance(config: RunConfig) -> dict[str, Any]:
    from tessera_nordic.area import load_area_geometry
    from tessera_nordic.sources import build_sources

    geom = load_area_geometry(config.area)
    return {s.name: s.provenance() for s in build_sources(config, tuple(geom.bounds))}


def _aggregate(config: RunConfig, run_dir: Path, workers: int | None) -> None:
    from tessera_nordic.pipeline.aggregate import run_aggregate

    run_aggregate(config, run_dir, workers=workers)


@app.command()
def plan(config_path: ConfigArg) -> None:
    """Count the tiles in the area and estimate the data volume (no download)."""
    from tessera_nordic.area import load_area_geometry, tiles_for_geometry

    load_dotenv(override=False)
    setup_logging()
    config = load_config(config_path)
    tiles = _guard("plan", lambda: tiles_for_geometry(load_area_geometry(config.area)))
    log.info(
        "%s: %d tiles of 0.1 deg (upper bound; open-sea tiles carry no data): ~%.2f TB of "
        "tile data (~%.0f MB/tile decoded), ~%.1f TB network transfer (~%.0f MB/tile measured)",
        config.area.name,
        len(tiles),
        len(tiles) * _BYTES_PER_TILE / 1e12,
        _BYTES_PER_TILE / 1e6,
        len(tiles) * _NETWORK_BYTES_PER_TILE / 1e12,
        _NETWORK_BYTES_PER_TILE / 1e6,
    )


@app.command()
def aggregate(
    config_path: ConfigArg, run_dir: RunDirOpt = None, workers: WorkersOpt = None
) -> None:
    """Stage 1: stream tiles and write per-tile cell statistics (resumable)."""
    config, rd = _prepare(config_path, run_dir, new=True)
    write_run_record(
        rd,
        config,
        config_path,
        "aggregate",
        _guard("provenance", lambda: _source_provenance(config)),
    )
    _guard("aggregate", lambda: _aggregate(config, rd, workers))


@app.command()
def merge(
    config_path: ConfigArg,
    run_dir: RunDirOpt = None,
    allow_incomplete: Annotated[
        bool, typer.Option(help="Merge even if tiles are missing.")
    ] = False,
) -> None:
    """Stage 2: combine tile statistics into per-cell mean/std (GeoParquet)."""
    from tessera_nordic.pipeline.merge import run_merge

    config, rd = _prepare(config_path, run_dir, new=False)
    write_run_record(rd, config, config_path, "merge")
    _guard("merge", lambda: run_merge(config, rd, allow_incomplete))


@app.command()
def cluster(config_path: ConfigArg, run_dir: RunDirOpt = None) -> None:
    """Stage 3: L2-normalise (optional), PCA and cluster for each k."""
    from tessera_nordic.pipeline.cluster import run_cluster

    config, rd = _prepare(config_path, run_dir, new=False)
    write_run_record(rd, config, config_path, "cluster")
    _guard("cluster", lambda: run_cluster(config, rd))


@app.command()
def evaluate(config_path: ConfigArg, run_dir: RunDirOpt = None) -> None:
    """Stage 4: metrics per k (silhouette, DB, CH, inertia, seed stability) and plots."""
    from tessera_nordic.pipeline.evaluate import run_evaluate

    config, rd = _prepare(config_path, run_dir, new=False)
    write_run_record(rd, config, config_path, "evaluate")
    _guard("evaluate", lambda: run_evaluate(config, rd))


@app.command("map")
def map_(config_path: ConfigArg, run_dir: RunDirOpt = None) -> None:
    """Stage 5: cluster maps as Cloud-Optimized GeoTIFF (grid CRS) and PNG quick-looks."""
    from tessera_nordic.pipeline.mapping import run_map

    config, rd = _prepare(config_path, run_dir, new=False)
    write_run_record(rd, config, config_path, "map")
    _guard("map", lambda: run_map(config, rd))


@app.command()
def compare(config_path: ConfigArg, run_dir: RunDirOpt = None) -> None:
    """Stage 6: compare groups (e.g. countries) over the shared clusters."""
    from tessera_nordic.pipeline.compare import run_compare

    config, rd = _prepare(config_path, run_dir, new=False)
    write_run_record(rd, config, config_path, "compare")
    _guard("compare", lambda: run_compare(config, rd))


@app.command()
def run(config_path: ConfigArg, run_dir: RunDirOpt = None, workers: WorkersOpt = None) -> None:
    """Run all stages (aggregate .. map, plus compare if configured) in one run directory."""
    from tessera_nordic.pipeline.cluster import run_cluster
    from tessera_nordic.pipeline.evaluate import run_evaluate
    from tessera_nordic.pipeline.mapping import run_map
    from tessera_nordic.pipeline.merge import run_merge

    config, rd = _prepare(config_path, run_dir, new=True)
    write_run_record(
        rd, config, config_path, "run", _guard("provenance", lambda: _source_provenance(config))
    )
    start = time.monotonic()
    _guard("aggregate", lambda: _aggregate(config, rd, workers))
    _guard("merge", lambda: run_merge(config, rd))
    _guard("cluster", lambda: run_cluster(config, rd))
    _guard("evaluate", lambda: run_evaluate(config, rd))
    _guard("map", lambda: run_map(config, rd))
    if config.compare is not None:
        from tessera_nordic.pipeline.compare import run_compare

        _guard("compare", lambda: run_compare(config, rd))
    log.info("Pipeline finished in %.1f min; outputs in %s", (time.monotonic() - start) / 60, rd)


def main() -> None:  # pragma: no cover
    """Entry point for ``python -m tessera_nordic``."""
    app()
