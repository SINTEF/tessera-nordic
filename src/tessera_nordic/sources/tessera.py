"""TESSERA embeddings as a feature source.

All direct use of the ``geotessera`` library is confined to the two reader classes in
this module, so an upstream API change only touches this file. See
``docs/decisions.md`` for what the API returns and why it is used this way.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from affine import Affine
from shapely.geometry.base import BaseGeometry

from tessera_nordic.area import TILE_DEG, tiles_for_geometry
from tessera_nordic.config import TesseraConfig
from tessera_nordic.grid import Grid, transformer
from tessera_nordic.sources.base import FeatureSource, WorkUnit
from tessera_nordic.stats import CellStats, accumulate

log = logging.getLogger(__name__)

N_BANDS = 128
WGS84 = "EPSG:4326"


@dataclass
class RawTile:
    """One tile as delivered upstream, still quantised.

    Attributes:
        quantized: ``(H, W, B)`` int8 embeddings (or already-float values with
            ``scales=None``).
        scales: ``(H, W)`` float32 per-pixel scales; non-finite marks water / no data.
        transform: Affine transform of the pixel grid (native UTM).
        crs: CRS of the pixel grid, e.g. ``"EPSG:32632"``.
        landmask: Optional ``(H, W)`` uint8 land mask (0 = water).
    """

    quantized: np.ndarray
    scales: np.ndarray | None
    transform: Affine
    crs: str
    landmask: np.ndarray | None = None


def tile_cell_stats(
    raw: RawTile, lon: float, lat: float, grid: Grid, apply_landmask: bool = True
) -> CellStats:
    """Reduce one tile to per-cell statistics on ``grid``.

    Steps: drop pixels with non-finite scales (and, if requested, land-mask water);
    keep only pixels whose centre lies in the tile's own half-open 0.1-degree box (tile
    arrays are UTM envelopes and overlap their neighbours); project pixel centres to the
    grid CRS; dequantise band by band while accumulating count, sum and sum of squares.

    Args:
        raw: The tile data.
        lon: Tile-centre longitude.
        lat: Tile-centre latitude.
        grid: Target grid.
        apply_landmask: Use ``raw.landmask`` (if present) to drop water pixels.

    Returns:
        Statistics for every cell that received at least one valid pixel.
    """
    q = raw.quantized
    if q.ndim != 3:
        raise ValueError(f"expected (H, W, B) embeddings, got shape {q.shape}")
    height, width, n_bands = q.shape
    valid = np.ones((height, width), dtype=bool)
    if raw.scales is not None:
        if raw.scales.shape != (height, width):
            raise ValueError(f"scales shape {raw.scales.shape} != {(height, width)}")
        valid &= np.isfinite(raw.scales)
    if apply_landmask and raw.landmask is not None:
        if raw.landmask.shape != (height, width):
            raise ValueError(f"landmask shape {raw.landmask.shape} != {(height, width)}")
        valid &= raw.landmask != 0

    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        return CellStats.empty(n_bands)
    t = raw.transform
    px = t.a * (cols + 0.5) + t.b * (rows + 0.5) + t.c
    py = t.d * (cols + 0.5) + t.e * (rows + 0.5) + t.f

    # Ownership: the pixel centre must fall inside this tile's own lon/lat box.
    plon, plat = transformer(raw.crs, WGS84).transform(px, py)
    lon0 = lon - TILE_DEG / 2
    lat0 = lat - TILE_DEG / 2
    eps = 1e-9
    owned = (
        (plon >= lon0 - eps)
        & (plon < lon0 + TILE_DEG - eps)
        & (plat >= lat0 - eps)
        & (plat < lat0 + TILE_DEG - eps)
    )
    rows, cols, px, py = rows[owned], cols[owned], px[owned], py[owned]
    if rows.size == 0:
        return CellStats.empty(n_bands)

    gx, gy = transformer(raw.crs, grid.crs).transform(px, py)
    cell_id = grid.cell_ids(gx, gy)

    flat = rows * width + cols
    qv = q.reshape(-1, n_bands)[flat]
    if raw.scales is not None:
        sv = raw.scales.reshape(-1)[flat].astype(np.float32)

        def feature(i: int) -> np.ndarray:
            return qv[:, i].astype(np.float32) * sv
    else:

        def feature(i: int) -> np.ndarray:
            return qv[:, i].astype(np.float32)

    return accumulate(cell_id, n_bands, feature)


class TileReader(Protocol):
    """Fetches raw tiles from one TESSERA distribution."""

    def available(self, tiles: list[tuple[float, float]], year: int) -> list[tuple[float, float]]:
        """Subset of ``tiles`` that exist upstream (best effort)."""
        ...

    def read(self, lon: float, lat: float, year: int) -> RawTile | None:
        """Fetch one tile; ``None`` if it has no data."""
        ...

    def cleanup(self, lon: float, lat: float, year: int) -> None:
        """Delete any raw files kept for the tile."""
        ...

    def provenance(self) -> dict[str, Any]:
        """Dataset/model versions."""
        ...


class ZarrTileReader:
    """Reads tiles from the TESSERA Zarr store (``geotessera.GeoTesseraZarr``)."""

    def __init__(self, cfg: TesseraConfig, cache_dir: Path | None) -> None:
        """Create the reader; the store is opened lazily on first use.

        Args:
            cfg: TESSERA settings.
            cache_dir: Local chunk cache, or ``None`` to stream without caching.
        """
        self.cfg = cfg
        self.cache_dir = cache_dir
        self._gt: Any = None

    @property
    def gt(self) -> Any:
        """The underlying ``GeoTesseraZarr`` instance."""
        if self._gt is None:
            from geotessera import GeoTesseraZarr
            from geotessera.registry import zarr_store_url

            url = self.cfg.store_url or zarr_store_url(
                self.cfg.dataset_version, self.cfg.dataset_variant
            )
            self._gt = GeoTesseraZarr(url, cache_dir=self.cache_dir)
        return self._gt

    def available(self, tiles: list[tuple[float, float]], year: int) -> list[tuple[float, float]]:
        """The store has no tile listing; tiles over open sea simply yield no pixels."""
        years = self.gt.years
        if year not in years:
            raise ValueError(f"year {year} not in store; available {years}")
        return tiles

    def read(self, lon: float, lat: float, year: int) -> RawTile | None:
        """Read the tile's lon/lat box (as its UTM envelope), quantised."""
        h = TILE_DEG / 2
        bbox = (lon - h, lat - h, lon + h, lat + h)
        try:
            emb, scales, transform, crs = self.gt.read_region_quantized(bbox, year)
        except IndexError:  # region outside the zone grid
            return None
        return RawTile(np.asarray(emb), np.asarray(scales), Affine(*transform[:6]), str(crs))

    def cleanup(self, lon: float, lat: float, year: int) -> None:
        """Nothing to do: the Zarr backend streams (cache only with ``keep_raw``)."""

    def provenance(self) -> dict[str, Any]:
        """Store URL, dataset, model and build versions."""
        ds = self.gt.dataset
        return {
            "backend": "zarr",
            "store_url": self.gt.url,
            "dataset_version": getattr(ds, "version", None),
            "dataset_variant": getattr(ds, "variant", None),
            "dataset_description": getattr(ds, "description", None),
            "model_version": self.gt.model_version,
            "store_build_version": self.gt.build_version,
            "store_years": list(self.gt.years),
        }


class NpyTileReader:
    """Downloads per-tile NPY files with ``geotessera.GeoTessera`` (deprecated upstream)."""

    def __init__(
        self, cfg: TesseraConfig, embeddings_dir: Path, bbox: tuple[float, float, float, float]
    ) -> None:
        """Create the reader; the registry is loaded lazily on first use.

        Args:
            cfg: TESSERA settings.
            embeddings_dir: Where tiles are downloaded (``TESSERA_DATA_DIR``).
            bbox: Area bounds (lon/lat), to load only the relevant part of the manifest.
        """
        self.cfg = cfg
        self.embeddings_dir = embeddings_dir
        self.bbox = bbox
        self._gt: Any = None

    @property
    def gt(self) -> Any:
        """The underlying ``GeoTessera`` instance."""
        if self._gt is None:
            from geotessera import GeoTessera

            self.embeddings_dir.mkdir(parents=True, exist_ok=True)
            self._gt = GeoTessera(
                dataset_version=self.cfg.dataset_version,
                dataset_variant=self.cfg.dataset_variant,
                embeddings_dir=self.embeddings_dir,
                cache_dir=self.embeddings_dir / "registry-cache",
                bbox=self.bbox,
            )
        return self._gt

    def available(self, tiles: list[tuple[float, float]], year: int) -> list[tuple[float, float]]:
        """Keep tiles listed in the NPY manifest for ``year``."""
        listed = {
            (round(lo, 2), round(la, 2))
            for _, lo, la in self.gt.registry.iter_tiles_in_region(self.bbox, year)
        }
        return [t for t in tiles if t in listed]

    def _paths(self, lon: float, lat: float, year: int) -> list[Path]:
        from geotessera.registry import (
            EMBEDDINGS_DIR_NAME,
            LANDMASKS_DIR_NAME,
            tile_to_embedding_paths,
            tile_to_landmask_filename,
        )

        emb, scales = tile_to_embedding_paths(lon, lat, year)
        base = self.embeddings_dir
        return [
            base / EMBEDDINGS_DIR_NAME / emb,
            base / EMBEDDINGS_DIR_NAME / scales,
            base / LANDMASKS_DIR_NAME / tile_to_landmask_filename(lon, lat),
        ]

    def read(self, lon: float, lat: float, year: int) -> RawTile | None:
        """Download (if needed) and load the tile's embeddings, scales and land mask."""
        import rasterio

        reg = self.gt.registry
        emb_path = reg.fetch(year=year, lon=lon, lat=lat, is_scales=False)
        scales_path = reg.fetch(year=year, lon=lon, lat=lat, is_scales=True)
        mask_path = reg.fetch_landmask(lon=lon, lat=lat)
        quantized = np.load(emb_path, mmap_mode="r")
        scales = np.load(scales_path)
        with rasterio.open(mask_path) as src:
            landmask = src.read(1)
            crs = src.crs.to_string()
            transform = src.transform
        return RawTile(quantized, scales, transform, crs, landmask)

    def cleanup(self, lon: float, lat: float, year: int) -> None:
        """Delete the tile's NPY files and land mask, and its now-empty directory."""
        paths = self._paths(lon, lat, year)
        for p in paths:
            p.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            paths[0].parent.rmdir()

    def provenance(self) -> dict[str, Any]:
        """Dataset version/variant and library version."""
        return {
            "backend": "npy",
            "dataset_version": self.gt.dataset_version,
            "dataset_variant": self.gt.dataset_variant,
            "embeddings_subdir": self.gt.embeddings_subdir,
        }


class TesseraSource(FeatureSource):
    """TESSERA annual 128-d embeddings, aggregated from 10 m pixels to grid cells."""

    name = "tessera"
    n_features = N_BANDS

    def __init__(
        self,
        cfg: TesseraConfig,
        year: int,
        grid: Grid,
        reader: TileReader,
    ) -> None:
        """Create the source.

        Args:
            cfg: TESSERA settings.
            year: Embedding year.
            grid: Target grid (part of the cache key).
            reader: Tile reader (Zarr, NPY, or a fake in tests).
        """
        self.cfg = cfg
        self.year = year
        self.grid = grid
        self.reader = reader

    def cache_key(self) -> str:
        """E.g. ``tessera_v1.1-dclimate_2024_EPSG3035_1000m_lm1``."""
        crs = self.grid.crs.replace(":", "")
        return (
            f"tessera_{self.cfg.dataset_version}-{self.cfg.dataset_variant}_{self.year}_"
            f"{crs}_{self.grid.cell_size}m_lm{int(self.cfg.apply_landmask)}"
        )

    def list_units(self, area: BaseGeometry) -> list[WorkUnit]:
        """One unit per 0.1-degree tile that overlaps ``area`` and exists upstream."""
        tiles = tiles_for_geometry(area)
        available = self.reader.available(tiles, self.year)
        if len(available) < len(tiles):
            log.warning(
                "%d of %d tiles in the area are not published for %s %s/%s",
                len(tiles) - len(available),
                len(tiles),
                self.year,
                self.cfg.dataset_version,
                self.cfg.dataset_variant,
            )
        return [
            WorkUnit(
                f"{self.year}_{lon:.2f}_{lat:.2f}",
                {"lon": lon, "lat": lat, "sample_key": f"{lon:.2f}_{lat:.2f}"},
            )
            for lon, lat in available
        ]

    def process_unit(self, unit: WorkUnit, grid: Grid) -> CellStats:
        """Fetch a tile and reduce it to cell statistics."""
        lon, lat = float(unit.params["lon"]), float(unit.params["lat"])
        raw = self.reader.read(lon, lat, self.year)
        if raw is None:
            return CellStats.empty(self.n_features)
        return tile_cell_stats(raw, lon, lat, grid, self.cfg.apply_landmask)

    def cleanup_unit(self, unit: WorkUnit) -> None:
        """Delete raw files of the tile."""
        self.reader.cleanup(float(unit.params["lon"]), float(unit.params["lat"]), self.year)

    def discard_unit(self, unit: WorkUnit) -> None:
        """Delete raw files so a retry downloads them again."""
        self.cleanup_unit(unit)

    def provenance(self) -> dict[str, Any]:
        """Dataset settings plus what the reader reports."""
        info: dict[str, Any] = {
            "source": self.name,
            "year": self.year,
            "configured_dataset_version": self.cfg.dataset_version,
            "configured_dataset_variant": self.cfg.dataset_variant,
            "apply_landmask": self.cfg.apply_landmask,
        }
        try:
            info.update(self.reader.provenance())
        except Exception as exc:  # provenance must never break a run
            log.warning("Could not read TESSERA provenance: %s", exc)
            info["provenance_error"] = str(exc)
        return info
