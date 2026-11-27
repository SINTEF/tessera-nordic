"""Stage 2 - merge: combine per-tile statistics into one per-cell GeoParquet table."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from tessera_nordic import paths
from tessera_nordic.area import load_area_geometry
from tessera_nordic.config import RunConfig
from tessera_nordic.grid import Grid, transformer
from tessera_nordic.pipeline.aggregate import SUMMARY_FILE, unit_path
from tessera_nordic.stats import CELL_ID, CellStats, finalize, merge_frames

log = logging.getLogger(__name__)

CELLS_FILE = "cells.parquet"


def mean_col(source: str, i: int) -> str:
    """Merged column name of the mean of feature ``i`` of ``source``."""
    return f"{source}_mean_{i:03d}"


def std_col(source: str, i: int) -> str:
    """Merged column name of the standard deviation of feature ``i`` of ``source``."""
    return f"{source}_std_{i:03d}"


def count_col(source: str) -> str:
    """Merged column name of the valid-pixel count of ``source``."""
    return f"{source}_count"


def source_table(
    name: str, stats_dir: Path, unit_ids: list[str], allow_incomplete: bool = False
) -> pd.DataFrame:
    """Per-cell count, mean and std of one source from its per-unit files.

    Args:
        name: Source name (column prefix).
        stats_dir: The source's statistics directory.
        unit_ids: Units that make up the area.
        allow_incomplete: Continue if some unit files are missing.
    """
    files = [unit_path(stats_dir, u) for u in unit_ids]
    missing = [f for f in files if not f.exists()]
    if missing:
        msg = f"{name}: {len(missing)} of {len(files)} tile statistics missing (e.g. {missing[0]})"
        if not allow_incomplete:
            raise FileNotFoundError(msg + "; rerun aggregate or pass --allow-incomplete")
        log.warning(msg)
    frames = [pd.read_parquet(f) for f in files if f.exists()]
    merged = merge_frames(frames)
    n_features = sum(c.startswith("sum_") for c in merged.columns)
    stats = CellStats.from_frame(merged, n_features)
    mean, std = finalize(stats)
    out: dict[str, Any] = {CELL_ID: stats.cell_id, count_col(name): stats.count}
    out.update({mean_col(name, i): mean[:, i] for i in range(n_features)})
    out.update({std_col(name, i): std[:, i] for i in range(n_features)})
    log.info("%s: merged %d tile files into %d cells", name, len(frames), len(stats))
    return pd.DataFrame(out)


def run_merge(config: RunConfig, run_dir: Path, allow_incomplete: bool = False) -> Path:
    """Merge all sources into ``run_dir/cells.parquet`` (GeoParquet, EPSG grid CRS).

    Columns: ``cell_id``, ``cell_code``, ``x``, ``y`` (cell centre), per source
    ``<name>_count``, ``<name>_mean_000..`` and ``<name>_std_000..``, and a point geometry.

    Args:
        config: Run configuration.
        run_dir: Run directory containing ``aggregate_summary.json``.
        allow_incomplete: Merge even if some tiles are missing.

    Returns:
        Path of the written file.
    """
    summary_path = run_dir / SUMMARY_FILE
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path} not found; run aggregate first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    grid = Grid(config.grid.crs, config.grid.cell_size)

    table: pd.DataFrame | None = None
    for name, info in summary["sources"].items():
        # Resolve from TN_DATA_DIR, not the recorded absolute path, so a run directory
        # stays usable across machines and between host and container.
        sdir = paths.data_dir() / "stats" / info["cache_key"]
        df = source_table(name, sdir, info["units"], allow_incomplete)
        n0 = len(df)
        df = df[df[count_col(name)] >= config.merge.min_pixel_count]
        log.info(
            "%s: dropped %d cells with < %d valid pixels",
            name,
            n0 - len(df),
            config.merge.min_pixel_count,
        )
        table = df if table is None else table.merge(df, on=CELL_ID, how="inner")
    if table is None or table.empty:
        raise ValueError("no cells left after merging; check the area and min_pixel_count")

    cell_id = table[CELL_ID].to_numpy(np.int64)
    x, y = grid.centres(cell_id)
    if config.merge.clip_to_area:
        geom = load_area_geometry(config.area)
        lon, lat = transformer(grid.crs, "EPSG:4326").transform(x, y)
        inside = shapely.contains_xy(geom, lon, lat)
        log.info("Dropped %d cells whose centre is outside the area", int((~inside).sum()))
        table, x, y, cell_id = table[inside], x[inside], y[inside], cell_id[inside]

    table = table.reset_index(drop=True)
    table.insert(1, "cell_code", grid.cell_codes(cell_id))
    table.insert(2, "x", x)
    table.insert(3, "y", y)
    gdf = gpd.GeoDataFrame(table, geometry=gpd.points_from_xy(x, y), crs=grid.crs)
    out = run_dir / CELLS_FILE
    gdf.to_parquet(out, compression="zstd")
    log.info("Wrote %d cells to %s", len(gdf), out)
    return out
