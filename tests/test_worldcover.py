"""WorldCover class fractions and the merge water filter, without network access."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from tessera_nordic import paths
from tessera_nordic.config import RunConfig, WorldCoverConfig
from tessera_nordic.grid import Grid
from tessera_nordic.pipeline.aggregate import run_aggregate, sample_units
from tessera_nordic.pipeline.merge import mean_col, run_merge
from tessera_nordic.sources.worldcover import (
    CLASSES,
    WATER_INDEX,
    WorldCoverSource,
    super_tile,
    window_stats,
)

from .conftest import FakeReader, make_source

N = 1200  # pixels per 0.1-degree tile


class FakeWorldCover(WorldCoverSource):
    """West half of every tile is water (80), east half tree cover (10)."""

    def read_window(self, lon: float, lat: float) -> np.ndarray:
        a = np.full((N, N), 10, np.uint8)
        if lon < 10.4:
            a[:, : N // 2] = 80
        return a


def test_super_tile_names() -> None:
    assert super_tile(10.3, 63.4) == ("N63E009", 9.0, 63.0)
    assert super_tile(-0.05, -0.05) == ("S03W003", -3.0, -3.0)


def test_window_stats_fractions() -> None:
    grid = Grid("EPSG:3035", 1000)
    classes = np.zeros((N, N), np.uint8)  # 0 = no data, dropped
    classes[:, :600] = 80
    classes[:, 600:900] = 10
    st = window_stats(classes, 10.3, 63.4, grid)
    frac = st.sums.sum(0) / st.count.sum()
    assert st.count.sum() == N * 900
    assert frac[WATER_INDEX] == pytest.approx(2 / 3)
    assert frac[list(CLASSES).index(10)] == pytest.approx(1 / 3)
    assert np.allclose(st.sums.sum(1), st.count)  # one-hot: fractions sum to 1


def test_water_filter_in_merge(env_dirs, small_config: RunConfig) -> None:
    config = RunConfig.model_validate(
        small_config.model_dump()
        | {
            "sources": {"worldcover": {}},
            "merge": {"min_pixel_count": 50, "max_water_fraction": 0.2},
        }
    )
    grid = Grid(config.grid.crs, config.grid.cell_size)
    wc = FakeWorldCover(WorldCoverConfig(), grid)
    run_dir = paths.new_run_dir(config.run_name)
    run_aggregate(config, run_dir, sources=[make_source(config, FakeReader()), wc])
    run_merge(config, run_dir)
    cells = pd.read_parquet(run_dir / "cells.parquet")
    water = cells[mean_col("worldcover", WATER_INDEX)]
    assert len(cells) > 20
    assert (water <= 0.2).all()

    no_filter = RunConfig.model_validate(config.model_dump() | {"merge": {"min_pixel_count": 50}})
    run_merge(no_filter, run_dir)
    joined = pd.read_parquet(run_dir / "cells.parquet")
    assert (joined[water.name] > 0.2).any()


def test_one_tile_gives_the_same_cells_in_both_sources(small_config: RunConfig) -> None:
    # Tile keys are tile centres for both sources; a half-tile offset would show here.
    grid = Grid("EPSG:3035", 1000)
    tess = make_source(small_config, FakeReader())
    wc = FakeWorldCover(WorldCoverConfig(), grid)
    box_area = box(10.3, 63.3, 10.4, 63.4)
    unit_t = tess.list_units(box_area)[0]
    unit_w = wc.list_units(box_area)[0]
    assert unit_t.params["sample_key"] == unit_w.params["sample_key"]
    ct = set(tess.process_unit(unit_t, grid).cell_id)
    cw = set(wc.process_unit(unit_w, grid).cell_id)
    assert len(ct & cw) / len(ct | cw) > 0.85


def test_sources_sample_the_same_tiles(small_config: RunConfig) -> None:

    grid = Grid("EPSG:3035", 1000)
    area = box(5, 58, 15, 66)
    tess = make_source(small_config, FakeReader()).list_units(area)
    wc = WorldCoverSource(WorldCoverConfig(), grid).list_units(area)
    pick = lambda us: {u.params["sample_key"] for u in sample_units(us, 0.05, 42)}  # noqa: E731
    assert pick(tess) == pick(wc)


def test_water_filter_needs_worldcover(small_config: RunConfig) -> None:
    with pytest.raises(ValueError, match="worldcover"):
        RunConfig.model_validate(small_config.model_dump() | {"merge": {"max_water_fraction": 0.2}})
