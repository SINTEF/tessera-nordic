"""Unit tests for the grid, area tiling and sufficient statistics."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

from tessera_nordic.area import tiles_for_geometry
from tessera_nordic.grid import Grid
from tessera_nordic.stats import CellStats, accumulate, finalize, merge_frames


def test_cell_id_roundtrip_including_negative() -> None:
    ix = np.array([0, 1, 4321, -5, 2**20])
    iy = np.array([0, -1, 3210, 7, -(2**20)])
    out_ix, out_iy = Grid.decode(Grid.encode(ix, iy))
    np.testing.assert_array_equal(out_ix, ix)
    np.testing.assert_array_equal(out_iy, iy)


def test_cells_are_aligned_to_multiples_of_cell_size() -> None:
    grid = Grid("EPSG:3035", 1000)
    x = np.array([4321000.0, 4321999.9, 4322000.0])
    y = np.array([3210000.0, 3210500.0, 3210999.0])
    ids = grid.cell_ids(x, y)
    assert ids[0] == ids[1] != ids[2]
    cx, cy = grid.centres(ids[:1])
    assert (cx[0], cy[0]) == (4321500.0, 3210500.0)
    assert grid.cell_codes(ids[:1])[0] == "1kmE4321N3210"
    assert Grid("EPSG:3035", 100).cell_codes(Grid("EPSG:3035", 100).cell_ids(x[:1], y[:1]))[0] == (
        "100mE43210N32100"
    )


def test_raster_for_places_cells_north_up() -> None:
    grid = Grid("EPSG:3035", 1000)
    ids = Grid.encode(np.array([10, 12, 10]), np.array([20, 20, 22]))
    transform, h, w, rows, cols = grid.raster_for(ids)
    assert (h, w) == (3, 3)
    assert (transform.c, transform.f) == (10000.0, 23000.0)
    np.testing.assert_array_equal(rows, [2, 2, 0])
    np.testing.assert_array_equal(cols, [0, 2, 0])


def test_tile_aligned_bbox_selects_exactly_its_tiles() -> None:
    from shapely.geometry import box

    tiles = tiles_for_geometry(box(10.3, 63.3, 10.5, 63.5))
    assert tiles == [(10.35, 63.35), (10.35, 63.45), (10.45, 63.35), (10.45, 63.45)]


def test_polygon_tiles_skip_boxes_outside() -> None:
    triangle = Polygon([(10.0, 63.0), (10.3, 63.0), (10.0, 63.3)])
    tiles = tiles_for_geometry(triangle)
    assert (10.05, 63.05) in tiles
    assert (10.25, 63.25) not in tiles  # beyond the hypotenuse
    assert len(tiles) == 6  # 3 + 2 + 1 tiles strictly overlapping the triangle


def test_accumulate_matches_direct_computation() -> None:
    rng = np.random.default_rng(0)
    cell = rng.integers(0, 5, size=1000)
    values = rng.normal(size=(1000, 3))
    stats = accumulate(cell, 3, lambda i: values[:, i])
    mean, std = finalize(stats)
    for j, c in enumerate(stats.cell_id):
        sel = values[cell == c]
        assert stats.count[j] == len(sel)
        np.testing.assert_allclose(mean[j], sel.mean(axis=0), rtol=1e-5)
        np.testing.assert_allclose(std[j], sel.std(axis=0), rtol=1e-4)


def test_statistics_are_mergeable_across_partitions() -> None:
    """Splitting pixels (as tile borders do) and merging gives the same result."""
    rng = np.random.default_rng(1)
    cell = rng.integers(100, 110, size=2000)
    values = rng.normal(loc=3.0, size=(2000, 4))
    whole = accumulate(cell, 4, lambda i: values[:, i])
    parts = np.array_split(np.arange(2000), 3)
    frames = [accumulate(cell[p], 4, lambda i, p=p: values[p, i]).to_frame() for p in parts]
    merged = CellStats.from_frame(merge_frames(frames), 4)
    np.testing.assert_array_equal(merged.cell_id, whole.cell_id)
    np.testing.assert_array_equal(merged.count, whole.count)
    np.testing.assert_allclose(merged.sums, whole.sums)
    np.testing.assert_allclose(merged.sumsq, whole.sumsq)


def test_empty_statistics_roundtrip() -> None:
    empty = CellStats.empty(4)
    assert len(CellStats.from_frame(empty.to_frame(), 4)) == 0
    with pytest.raises(ValueError):
        merge_frames([empty.to_frame()])
