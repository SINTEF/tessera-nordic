"""Stage 5 - map: cluster labels as Cloud-Optimized GeoTIFFs plus quick-look PNGs."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rasterio.io import MemoryFile
from rasterio.shutil import copy as rio_copy

from tessera_nordic.config import RunConfig
from tessera_nordic.grid import Grid
from tessera_nordic.pipeline.cluster import CLUSTER_DIR, LABELS_FILE, label_col

log = logging.getLogger(__name__)

MAP_DIR = "map"
NODATA = 255


def palette(k: int) -> list[tuple[float, float, float]]:
    """``k`` distinct RGB colours (tab10/tab20, then evenly spaced hues)."""
    if k <= 10:
        cmap = plt.get_cmap("tab10")
    elif k <= 20:
        cmap = plt.get_cmap("tab20")
    else:
        cmap = plt.get_cmap("hsv", k + 1)
    return [tuple(float(c) for c in cmap(i)[:3]) for i in range(k)]  # type: ignore[misc]


def label_raster(grid: Grid, cell_id: np.ndarray, labels: np.ndarray):
    """Rasterise per-cell labels onto the grid.

    Returns:
        ``(array, transform)`` with ``uint8`` labels and :data:`NODATA` elsewhere.
    """
    if labels.max() >= NODATA:
        raise ValueError(f"at most {NODATA} clusters can be mapped")
    transform, height, width, rows, cols = grid.raster_for(cell_id)
    arr = np.full((height, width), NODATA, dtype=np.uint8)
    arr[rows, cols] = labels.astype(np.uint8)
    return arr, transform


def write_cog(arr: np.ndarray, transform, crs: str, path: Path, k: int) -> None:
    """Write a single-band uint8 Cloud-Optimized GeoTIFF with an embedded colour table."""
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "height": arr.shape[0],
        "width": arr.shape[1],
        "crs": crs,
        "transform": transform,
        "nodata": NODATA,
    }
    colors = {i: (*tuple(int(255 * c) for c in rgb), 255) for i, rgb in enumerate(palette(k))}
    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(arr, 1)
            dst.write_colormap(1, colors)
            dst.update_tags(k=str(k), description=f"Cluster label (0..{k - 1}), {NODATA}=nodata")
        with mem.open() as src:
            rio_copy(
                src, str(path), driver="COG", compress="DEFLATE", overview_resampling="NEAREST"
            )


def write_quicklook(arr: np.ndarray, transform, k: int, title: str, path: Path, dpi: int) -> None:
    """PNG map with a categorical legend; axes in km of the grid CRS."""
    h, w = arr.shape
    left, top = transform.c, transform.f
    extent = (
        left / 1000,
        (left + w * transform.a) / 1000,
        (top + h * transform.e) / 1000,
        top / 1000,
    )
    masked = np.ma.masked_equal(arr, NODATA)
    aspect = h / max(w, 1)
    fig, ax = plt.subplots(figsize=(8, max(3.0, min(12.0, 8 * aspect))), constrained_layout=True)
    ax.imshow(
        masked,
        cmap=ListedColormap(palette(k)),
        vmin=-0.5,
        vmax=k - 0.5,
        extent=extent,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    handles = [Patch(color=c, label=str(i)) for i, c in enumerate(palette(k))]
    ax.legend(
        handles=handles,
        title="cluster",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=7,
        ncols=1 + k // 20,
    )
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def run_map(config: RunConfig, run_dir: Path) -> list[Path]:
    """Write ``map/clusters_k<k>.tif`` (COG) and ``map/clusters_k<k>.png`` for every k.

    Returns:
        Paths of the written GeoTIFFs.
    """
    out = run_dir / MAP_DIR
    out.mkdir(exist_ok=True)
    lab_df = pd.read_parquet(run_dir / CLUSTER_DIR / LABELS_FILE)
    grid = Grid(config.grid.crs, config.grid.cell_size)
    cell_id = lab_df["cell_id"].to_numpy(np.int64)
    written = []
    for k in config.cluster.k:
        col = label_col(k)
        if col not in lab_df:
            continue
        arr, transform = label_raster(grid, cell_id, lab_df[col].to_numpy())
        tif = out / f"clusters_k{k}.tif"
        write_cog(arr, transform, grid.crs, tif, k)
        title = (
            f"{config.run_name}: {config.area.name}, {config.year}, k={k} "
            f"({grid.cell_size} m cells)"
        )
        write_quicklook(arr, transform, k, title, out / f"clusters_k{k}.png", config.map.dpi)
        with rasterio.open(tif) as src:
            log.info("Wrote %s (%dx%d, %s)", tif.name, src.width, src.height, src.crs)
        written.append(tif)
    return written
