"""Feature sources. Register new sources in :func:`build_sources`."""

from __future__ import annotations

from tessera_nordic.config import RunConfig
from tessera_nordic.grid import Grid
from tessera_nordic.sources.base import FeatureSource, WorkUnit
from tessera_nordic.sources.tessera import NpyTileReader, TesseraSource, ZarrTileReader
from tessera_nordic.sources.worldcover import WorldCoverSource

__all__ = ["FeatureSource", "WorkUnit", "build_sources"]


def build_sources(
    config: RunConfig, area_bounds: tuple[float, float, float, float]
) -> list[FeatureSource]:
    """Instantiate every feature source enabled in ``config``.

    Args:
        config: Run configuration.
        area_bounds: Lon/lat bounds of the study area (lets readers limit what they load).
    """
    from tessera_nordic import paths

    grid = Grid(config.grid.crs, config.grid.cell_size)
    tcfg = config.sources.tessera
    reader: NpyTileReader | ZarrTileReader
    if tcfg.backend == "npy":
        reader = NpyTileReader(tcfg, paths.tessera_dir(), area_bounds)
    else:
        cache = paths.tessera_dir() / "zarr-cache" if config.aggregate.keep_raw else None
        reader = ZarrTileReader(tcfg, cache)
    sources: list[FeatureSource] = [TesseraSource(tcfg, config.year, grid, reader)]
    if config.sources.worldcover is not None:
        sources.append(WorldCoverSource(config.sources.worldcover, grid))
    return sources
