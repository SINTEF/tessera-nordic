"""Study-area geometry and the 0.1-degree TESSERA tile grid that covers it."""

from __future__ import annotations

import logging
import math

import geopandas as gpd
import shapely
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.prepared import prep

from tessera_nordic.config import AreaConfig, BoundaryConfig

log = logging.getLogger(__name__)

TILE_DEG = 0.1


def read_boundary(cfg: BoundaryConfig) -> gpd.GeoDataFrame:
    """Read a boundary file, keep the configured features and return them in EPSG:4326.

    Args:
        cfg: Boundary configuration.
    """
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
    log.info("Loaded boundary %s (%d features)", path.name, len(gdf))
    return gdf.to_crs("EPSG:4326")


def _clean(geoms: object, clip_bbox: tuple[float, float, float, float] | None) -> BaseGeometry:
    geom = shapely.union_all(shapely.make_valid(geoms))
    if clip_bbox is not None:
        geom = geom.intersection(box(*clip_bbox))
    return geom


def load_area_geometry(area: AreaConfig) -> BaseGeometry:
    """Return the study area as one (multi)polygon in EPSG:4326.

    Args:
        area: Area configuration (bbox or boundary file).
    """
    if area.bbox is not None:
        return box(*area.bbox)
    assert area.boundary is not None
    gdf = read_boundary(area.boundary)
    geom = _clean(gdf.geometry.values, area.boundary.clip_bbox)
    if geom.is_empty:
        raise ValueError("area geometry is empty after filtering/clipping")
    return geom


def load_group_geometries(cfg: BoundaryConfig) -> dict[str, BaseGeometry]:
    """One (multi)polygon per value of ``cfg.column`` (EPSG:4326), e.g. one per country.

    Args:
        cfg: Boundary configuration with ``column`` and ``values`` set.
    """
    if cfg.column is None or cfg.values is None:
        raise ValueError("group boundaries need `column` and `values`")
    gdf = read_boundary(cfg)
    groups = {}
    for value in cfg.values:
        sel = gdf[gdf[cfg.column].astype(str) == value]
        geom = _clean(sel.geometry.values, cfg.clip_bbox)
        if geom.is_empty:
            raise ValueError(f"group {value!r} has no geometry")
        groups[value] = geom
    return groups


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
