"""Group comparison over shared clusters, on synthetic tiles."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from tessera_nordic import paths
from tessera_nordic.config import RunConfig
from tessera_nordic.pipeline.aggregate import run_aggregate
from tessera_nordic.pipeline.cluster import run_cluster
from tessera_nordic.pipeline.compare import energy_distance, jsd, overlap, run_compare
from tessera_nordic.pipeline.merge import run_merge

from .conftest import FakeReader, make_source


def test_divergence_measures() -> None:
    p = np.array([10, 20, 30])
    assert jsd(p, p * 3) == pytest.approx(0.0, abs=1e-12)
    assert jsd(np.array([1, 0]), np.array([0, 1])) == pytest.approx(1.0)
    assert overlap(p, p) == pytest.approx(1.0)
    assert overlap(np.array([1, 0]), np.array([0, 1])) == pytest.approx(0.0)
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(400, 3)), rng.normal(size=(400, 3))
    assert energy_distance(a, b) < energy_distance(a, b + 3)


def test_compare_two_groups(env_dirs, small_config: RunConfig, tmp_path: Path) -> None:
    gpkg = tmp_path / "groups.gpkg"
    gpd.GeoDataFrame(
        {"name": ["W", "E"]},
        geometry=[box(10.0, 63.0, 10.4, 64.0), box(10.4, 63.0, 11.0, 64.0)],
        crs="EPSG:4326",
    ).to_file(gpkg)
    config = RunConfig.model_validate(
        small_config.model_dump()
        | {
            "compare": {
                "groups": {"path": str(gpkg), "column": "name", "values": ["W", "E"]},
                "block_size": 3000,
                "n_bootstrap": 60,
                "energy_sample": 200,
            }
        }
    )
    run_dir = paths.new_run_dir(config.run_name)
    run_aggregate(config, run_dir, sources=[make_source(config, FakeReader())])
    run_merge(config, run_dir)
    run_cluster(config, run_dir)
    summary = pd.read_csv(run_compare(config, run_dir))

    out = run_dir / "compare"
    shares = pd.read_csv(out / "cluster_shares.csv")
    assert set(shares.group) == {"W", "E"}
    for (_, _), d in shares.groupby(["k", "group"]):
        assert d.share.sum() == pytest.approx(1.0)
        assert (d.share_ci_low <= d.share + 1e-12).all() and (
            d.share <= d.share_ci_high + 1e-12
        ).all()
    assert list(summary.k) == [2, 3]
    assert ((summary.jsd >= 0) & (summary.jsd <= 1)).all()
    assert (summary.jsd_ci_low <= summary.jsd_ci_high).all()
    assert {
        "jsd_within_W_median",
        "jsd_within_E_median",
        "jsd_between_halves_median",
        "n_clusters_shared",
    } <= set(summary.columns)
    groups = pd.read_parquet(out / "cell_groups.parquet")
    assert set(groups.group) <= {"W", "E", ""}
    assert (groups.group == "W").sum() > 20 and (groups.group == "E").sum() > 20
    ed = pd.read_csv(out / "embedding_distance.csv")
    assert "W vs E" in set(ed.comparison)
    for name in ("shares_k3.png", "divergence_vs_k.png"):
        assert (out / name).stat().st_size > 1000
