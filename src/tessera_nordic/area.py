"""Study-area geometry and the 0.1-degree TESSERA tile grid that covers it."""

from __future__ import annotations

import logging
import math

import geopandas as gpd
import shapely
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.prepared import prep

from tessera_nordic.config import AreaConfig

log = logging.getLogger(__name__)

TILE_DEG = 0.1


def load_area_geometry(area: AreaConfig) -> BaseGeometry:
    """Return the study area as one (multi)polygon in EPSG:4326.

    Args:
        area: Area configuration (bbox or boundary file).
    """
    if area.bbox is not None:
        return box(*area.bbox)
    assert area.boundary is not None
    cfg = area.boundary
    path = cfg.resolved_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Boundary file {path} not found. See the README section 'Country boundaries' "
            "for where to download it."
        )
    gdf = gpd.read_file(path)
    if cfg.column is not None and cfg.values is not None:
        if cfg.column not in gdf.columns:
            raise KeyError(f"column {cfg.column!r} not in {path}; have {list(gdf.columns)}")
        gdf = gdf[gdf[cfg.column].astype(str).isin(cfg.values)]
        if gdf.empty:
            raise ValueError(f"no features in {path} with {cfg.column} in {cfg.values}")
    gdf = gdf.to_crs("EPSG:4326")
    geom = shapely.union_all(shapely.make_valid(gdf.geometry.values))
    if cfg.clip_bbox is not None:
        geom = geom.intersection(box(*cfg.clip_bbox))
    if geom.is_empty:
        raise ValueError("area geometry is empty after filtering/clipping")
    log.info("Loaded boundary %s (%d features)", path.name, len(gdf))
    return geom


def tile_box(lon: float, lat: float) -> BaseGeometry:
    """The 0.1-degree box of the tile centred at ``(lon, lat)``."""
    h = TILE_DEG / 2
    return box(lon - h, lat - h, lon + h, lat + h)


def tiles_for_geometry(geom: BaseGeometry) -> list[tuple[float, float]]:
    """Centres of all 0.1-degree tiles whose box overlaps ``geom`` with non-zero area.

    Tiles that only touch the geometry along an edge are excluded, so a bbox aligned
    to tile edges selects exactly the tiles inside it.

    Args:
        geom: Polygon(s) in EPSG:4326.

    Returns:
        Sorted list of ``(lon, lat)`` tile centres (rounded to 2 decimals).
    """
    min_lon, min_lat, max_lon, max_lat = geom.bounds
    i0 = math.floor(round(min_lon / TILE_DEG, 9))
    i1 = math.ceil(round(max_lon / TILE_DEG, 9))
    j0 = math.floor(round(min_lat / TILE_DEG, 9))
    j1 = math.ceil(round(max_lat / TILE_DEG, 9))
    prepared = prep(geom)
    tiles: list[tuple[float, float]] = []
    for i in range(i0, i1):
        for j in range(j0, j1):
            lon = round((i + 0.5) * TILE_DEG, 2)
            lat = round((j + 0.5) * TILE_DEG, 2)
            tb = tile_box(lon, lat)
            if not prepared.intersects(tb):
                continue
            if prepared.contains(tb) or geom.intersection(tb).area > 0:
                tiles.append((lon, lat))
    return sorted(tiles)
