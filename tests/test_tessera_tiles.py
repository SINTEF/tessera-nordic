"""Per-tile processing of (synthetic) TESSERA tiles."""

from __future__ import annotations

import numpy as np

from tessera_nordic.grid import Grid, transformer
from tessera_nordic.sources.tessera import tile_cell_stats
from tessera_nordic.stats import CellStats, finalize, merge_frames

from .conftest import N_BANDS, SCALE, synthetic_tile

GRID = Grid("EPSG:3035", 1000)


def _owned_pixels(tile, lon: float, lat: float) -> np.ndarray:
    h, w, _ = tile.quantized.shape
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    x = tile.transform.c + (cols + 0.5) * tile.transform.a
    y = tile.transform.f + (rows + 0.5) * tile.transform.e
    plon, plat = transformer(tile.crs, "EPSG:4326").transform(x, y)
    return (plon >= lon - 0.05) & (plon < lon + 0.05) & (plat >= lat - 0.05) & (plat < lat + 0.05)


def test_tiles_overlap_but_pixels_are_counted_once() -> None:
    a = synthetic_tile(10.35, 63.45)
    b = synthetic_tile(10.45, 63.45)
    # The UTM envelopes overlap, as real tiles do.
    assert a.transform.c + a.quantized.shape[1] * a.transform.a > b.transform.c

    sa = tile_cell_stats(a, 10.35, 63.45, GRID)
    sb = tile_cell_stats(b, 10.45, 63.45, GRID)
    expected = 0
    for tile, lon in ((a, 10.35), (b, 10.45)):
        owned = _owned_pixels(tile, lon, 63.45) & np.isfinite(tile.scales)
        expected += int(owned.sum())
    assert int(sa.count.sum() + sb.count.sum()) == expected


def test_dequantised_cell_means_match_direct_computation() -> None:
    tile = synthetic_tile(10.35, 63.35)
    stats = tile_cell_stats(tile, 10.35, 63.35, GRID)
    mean, std = finalize(stats)

    valid = _owned_pixels(tile, 10.35, 63.35) & np.isfinite(tile.scales)
    rows, cols = np.nonzero(valid)
    x = tile.transform.c + (cols + 0.5) * tile.transform.a
    y = tile.transform.f + (rows + 0.5) * tile.transform.e
    gx, gy = transformer(tile.crs, GRID.crs).transform(x, y)
    ids = GRID.cell_ids(gx, gy)
    values = tile.quantized[rows, cols].astype(np.float64) * SCALE
    j = len(stats) // 2
    sel = values[ids == stats.cell_id[j]]
    assert stats.count[j] == len(sel)
    np.testing.assert_allclose(mean[j], sel.mean(axis=0), rtol=1e-5)
    np.testing.assert_allclose(std[j], sel.std(axis=0), rtol=1e-4, atol=1e-6)
    assert stats.n_features == N_BANDS


def test_landmask_and_nonfinite_scales_are_excluded() -> None:
    tile = synthetic_tile(10.35, 63.35)
    base = tile_cell_stats(tile, 10.35, 63.35, GRID)

    tile.scales[:, :10] = np.inf  # "no data" columns
    tile.landmask = np.ones(tile.scales.shape, dtype=np.uint8)
    tile.landmask[:20, :] = 0  # "water" rows
    masked = tile_cell_stats(tile, 10.35, 63.35, GRID, apply_landmask=True)
    unmasked = tile_cell_stats(tile, 10.35, 63.35, GRID, apply_landmask=False)
    assert masked.count.sum() < unmasked.count.sum() < base.count.sum()


def test_cells_split_by_tile_border_merge_to_whole() -> None:
    """A cell straddling two tiles gets the same statistics as if read in one piece."""
    a = synthetic_tile(10.35, 63.45)
    b = synthetic_tile(10.45, 63.45)
    merged = CellStats.from_frame(
        merge_frames(
            [
                tile_cell_stats(a, 10.35, 63.45, GRID).to_frame(),
                tile_cell_stats(b, 10.45, 63.45, GRID).to_frame(),
            ]
        ),
        N_BANDS,
    )
    ids_a = set(tile_cell_stats(a, 10.35, 63.45, GRID).cell_id)
    ids_b = set(tile_cell_stats(b, 10.45, 63.45, GRID).cell_id)
    shared = ids_a & ids_b
    assert shared, "expected cells crossing the tile border"
    # A full 1 km cell holds 100 pixels of 100 m; shared cells should be (nearly) complete.
    counts = dict(zip(merged.cell_id, merged.count, strict=True))
    interior_like = [counts[c] for c in shared]
    assert max(interior_like) <= 100
    assert np.median(interior_like) >= 90


def test_empty_tile_gives_empty_stats() -> None:
    tile = synthetic_tile(10.35, 63.35)
    tile.scales[:] = np.nan
    assert len(tile_cell_stats(tile, 10.35, 63.35, GRID)) == 0
