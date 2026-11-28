"""ESA WorldCover 10 m land cover as a feature source: per-cell class fractions.

WorldCover is published as 3 x 3 degree Cloud-Optimised GeoTIFFs in EPSG:4326 with
1/12000-degree pixels, aligned to whole degrees. Each work unit is one 0.1-degree tile,
the same tiles as the TESSERA source (so a sampled run samples the same tiles for both),
and reads only that window over HTTP. Every pixel becomes a one-hot vector over
:data:`CLASSES`, so a cell's mean is its land-cover composition; ``water_fraction``
(class 80, permanent water bodies) is what ``merge.max_water_fraction`` filters on.

Licence: CC BY 4.0, (c) ESA WorldCover project / contains modified Copernicus
Sentinel data processed by the ESA WorldCover consortium.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import numpy as np
from shapely.geometry.base import BaseGeometry

from tessera_nordic.area import TILE_DEG, tiles_for_geometry
from tessera_nordic.config import WorldCoverConfig
from tessera_nordic.grid import Grid, transformer
from tessera_nordic.sources.base import FeatureSource, WorkUnit
from tessera_nordic.stats import CellStats, accumulate

log = logging.getLogger(__name__)

#: WorldCover class codes and names, in feature order (``worldcover_mean_000`` = tree cover).
CLASSES: dict[int, str] = {
    10: "tree_cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built_up",
    60: "bare_sparse",
    70: "snow_ice",
    80: "water",
    90: "herbaceous_wetland",
    95: "mangroves",
    100: "moss_lichen",
}
CODES = np.array(list(CLASSES))
WATER_INDEX = list(CLASSES).index(80)
PIXELS_PER_DEG = 12000
SUPER_TILE_DEG = 3
_GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "2",
}


def feature_name(i: int) -> str:
    """Land-cover class name of feature ``i``."""
    return list(CLASSES.values())[i]


def super_tile(lon: float, lat: float) -> tuple[str, float, float]:
    """Name and lower-left corner of the 3-degree file containing ``(lon, lat)``."""
    x0 = math.floor(round(lon, 6) / SUPER_TILE_DEG) * SUPER_TILE_DEG
    y0 = math.floor(round(lat, 6) / SUPER_TILE_DEG) * SUPER_TILE_DEG
    ns = f"N{y0:02d}" if y0 >= 0 else f"S{-y0:02d}"
    ew = f"E{x0:03d}" if x0 >= 0 else f"W{-x0:03d}"
    return f"{ns}{ew}", float(x0), float(y0)


def window_stats(classes: np.ndarray, lon: float, lat: float, grid: Grid) -> CellStats:
    """Class-fraction statistics of a ``(H, W)`` class window whose top-left is
    ``(lon, lat + H / PIXELS_PER_DEG)`` (north-up, 1/12000-degree pixels)."""
    h = classes.shape[0]
    valid = classes != 0
    if not valid.any():
        return CellStats.empty(len(CLASSES))
    rows, cols = np.nonzero(valid)
    plon = lon + (cols + 0.5) / PIXELS_PER_DEG
    plat = lat + (h - rows - 0.5) / PIXELS_PER_DEG
    x, y = transformer("EPSG:4326", grid.crs).transform(plon, plat)
    cell_id = grid.cell_ids(np.asarray(x), np.asarray(y))
    code = classes[rows, cols]
    return accumulate(cell_id, len(CLASSES), lambda i: (code == CODES[i]).astype(np.float32))


class WorldCoverSource(FeatureSource):
    """Per-cell fractions of the WorldCover land-cover classes."""

    name = "worldcover"
    n_features = len(CLASSES)

    def __init__(self, cfg: WorldCoverConfig, grid: Grid) -> None:
        self.cfg = cfg
        self.grid = grid
        self._open: dict[str, Any] = {}

    def cache_key(self) -> str:
        """E.g. ``worldcover_v200_2021_EPSG3035_1000m``."""
        crs = self.grid.crs.replace(":", "")
        return f"worldcover_{self.cfg.version}_{self.cfg.year}_{crs}_{self.grid.cell_size}m"

    def list_units(self, area: BaseGeometry) -> list[WorkUnit]:
        """One unit per 0.1-degree tile overlapping ``area`` (same tiles as TESSERA)."""
        return [
            WorkUnit(
                f"wc_{lon:.2f}_{lat:.2f}",
                {"lon": lon, "lat": lat, "sample_key": f"{lon:.2f}_{lat:.2f}"},
            )
            for lon, lat in tiles_for_geometry(area)
        ]

    def _url(self, name: str) -> str:
        return self.cfg.url_template.format(version=self.cfg.version, year=self.cfg.year, tile=name)

    def _dataset(self, name: str) -> Any:
        """Open (and keep open) one 3-degree file; ``None`` if it does not exist (open sea)."""
        import rasterio
        from rasterio.errors import RasterioIOError

        if name not in self._open:
            for k, v in _GDAL_ENV.items():
                os.environ.setdefault(k, v)
            try:
                self._open[name] = rasterio.open(self._url(name))
            except RasterioIOError as exc:
                if "404" not in str(exc):
                    raise
                log.debug("WorldCover file %s does not exist (open sea)", name)
                self._open[name] = None
        return self._open[name]

    def read_window(self, lon: float, lat: float) -> np.ndarray | None:
        """Class codes of the box ``[lon, lon+0.1) x [lat, lat+0.1)`` (lower-left corner)."""
        from rasterio.windows import Window

        name, x0, y0 = super_tile(lon + 1e-9, lat + 1e-9)
        ds = self._dataset(name)
        if ds is None:
            return None
        n = round(TILE_DEG * PIXELS_PER_DEG)
        col = round((lon - x0) * PIXELS_PER_DEG)
        row = round((y0 + SUPER_TILE_DEG - (lat + TILE_DEG)) * PIXELS_PER_DEG)
        return np.asarray(ds.read(1, window=Window(col, row, n, n)))

    def process_unit(self, unit: WorkUnit, grid: Grid) -> CellStats:
        """Read the tile window and reduce it to per-cell class fractions.

        Unit coordinates are tile *centres* (as for TESSERA); the window starts half a
        tile to the south-west.
        """
        west = float(unit.params["lon"]) - TILE_DEG / 2
        south = float(unit.params["lat"]) - TILE_DEG / 2
        classes = self.read_window(west, south)
        if classes is None:
            return CellStats.empty(self.n_features)
        return window_stats(classes, west, south, grid)

    def discard_unit(self, unit: WorkUnit) -> None:
        """Reopen files on retry, in case a connection went stale."""
        for ds in self._open.values():
            if ds is not None:
                ds.close()
        self._open.clear()

    def provenance(self) -> dict[str, Any]:
        """Dataset version and the class order of the features."""
        return {
            "source": self.name,
            "version": self.cfg.version,
            "year": self.cfg.year,
            "url_template": self.cfg.url_template,
            "features": {f"{self.name}_mean_{i:03d}": n for i, n in enumerate(CLASSES.values())},
            "licence": "CC BY 4.0 (ESA WorldCover)",
        }
