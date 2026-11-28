"""End-to-end pipeline on synthetic tiles: aggregate (resume, retry) -> merge -> cluster ->
evaluate -> map. No network access."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio

from tessera_nordic import paths
from tessera_nordic.config import RunConfig
from tessera_nordic.pipeline.aggregate import (
    AggregateIncompleteError,
    Manifest,
    run_aggregate,
    stats_dir,
)
from tessera_nordic.pipeline.cluster import run_cluster
from tessera_nordic.pipeline.evaluate import run_evaluate
from tessera_nordic.pipeline.mapping import NODATA, run_map
from tessera_nordic.pipeline.merge import run_merge
from tessera_nordic.provenance import write_run_record

from .conftest import FakeReader, make_source


def test_aggregate_resumes_and_skips_completed_tiles(env_dirs, small_config: RunConfig) -> None:
    reader = FakeReader()
    source = make_source(small_config, reader)
    run_dir = paths.new_run_dir(small_config.run_name)
    summary = run_aggregate(small_config, run_dir, sources=[source])
    info = summary["sources"]["tessera"]
    assert info["n_units"] == 4 and info["n_processed_now"] == 4
    assert len(reader.cleaned) == 4  # raw data removed (keep_raw: false)

    sdir = stats_dir(source)
    assert sdir.is_relative_to(env_dirs["TN_DATA_DIR"])
    assert len(Manifest(sdir).completed()) == 4

    reader2 = FakeReader()
    summary2 = run_aggregate(small_config, run_dir, sources=[make_source(small_config, reader2)])
    assert reader2.reads == []
    assert summary2["sources"]["tessera"]["n_skipped_done"] == 4


def test_keep_raw_skips_cleanup(env_dirs, small_config: RunConfig) -> None:
    config = small_config.model_copy(
        update={"aggregate": small_config.aggregate.model_copy(update={"keep_raw": True})}
    )
    reader = FakeReader()
    run_aggregate(config, paths.new_run_dir(config.run_name), sources=[make_source(config, reader)])
    assert reader.cleaned == []


def test_failed_downloads_are_retried(env_dirs, small_config: RunConfig) -> None:
    reader = FakeReader(fail_times={(10.35, 63.35): 2})  # max_retries = 2 -> 3rd attempt wins
    run_dir = paths.new_run_dir(small_config.run_name)
    summary = run_aggregate(small_config, run_dir, sources=[make_source(small_config, reader)])
    assert summary["sources"]["tessera"]["n_failed"] == 0
    assert reader.reads.count((10.35, 63.35)) == 3


def test_interrupted_run_continues(env_dirs, small_config: RunConfig) -> None:
    """A tile that keeps failing stops the run; a rerun processes only that tile."""
    run_dir = paths.new_run_dir(small_config.run_name)
    failing = FakeReader(fail_times={(10.45, 63.45): 10})
    with pytest.raises(AggregateIncompleteError):
        run_aggregate(small_config, run_dir, sources=[make_source(small_config, failing)])
    summary = json.loads((run_dir / "aggregate_summary.json").read_text())
    assert summary["sources"]["tessera"]["n_failed"] == 1

    healthy = FakeReader()
    run_aggregate(small_config, run_dir, sources=[make_source(small_config, healthy)])
    assert healthy.reads == [(10.45, 63.45)]


def test_full_pipeline_on_synthetic_tiles(
    env_dirs, small_config: RunConfig, tmp_path: Path
) -> None:
    run_dir = paths.new_run_dir(small_config.run_name)
    assert run_dir.parent.parent == env_dirs["TN_OUTPUT_DIR"]
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("placeholder: true\n")
    write_run_record(
        run_dir, small_config, cfg_file, "test", {"tessera": FakeReader().provenance()}
    )

    run_aggregate(small_config, run_dir, sources=[make_source(small_config, FakeReader())])

    cells_path = run_merge(small_config, run_dir)
    cells = gpd.read_parquet(cells_path)
    assert cells.crs.to_epsg() == 3035
    assert {
        "cell_id",
        "cell_code",
        "x",
        "y",
        "tessera_count",
        "tessera_mean_000",
        "tessera_std_007",
    } <= set(cells.columns)
    assert (cells["tessera_count"] >= small_config.merge.min_pixel_count).all()
    assert cells["cell_id"].is_unique
    assert np.all(cells["x"] % 1000 == 500)
    assert len(cells) > 100

    run_cluster(small_config, run_dir)
    labels = pd.read_parquet(run_dir / "cluster" / "labels.parquet")
    assert list(labels.columns) == ["cell_id", "label_k2", "label_k3"]
    assert (run_dir / "cluster" / "model_k3.joblib").exists()
    ev = pd.read_csv(run_dir / "cluster" / "pca_explained_variance.csv")
    assert len(ev) == 4 and ev["cumulative"].iloc[-1] <= 1.0 + 1e-9

    metrics = pd.read_csv(run_evaluate(small_config, run_dir))
    assert list(metrics["k"]) == [2, 3]
    for col in (
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "inertia",
        "stability_ari_mean",
    ):
        assert metrics[col].notna().all(), col
    # The synthetic field has three landscapes; k=3 should recover them stably.
    assert metrics.set_index("k").loc[3, "stability_ari_mean"] > 0.9
    assert (run_dir / "evaluate" / "metrics_vs_k.png").exists()

    tifs = run_map(small_config, run_dir)
    assert [t.name for t in tifs] == ["clusters_k2.tif", "clusters_k3.tif"]
    with rasterio.open(tifs[1]) as src:
        assert src.crs.to_epsg() == 3035
        assert src.res == (1000.0, 1000.0)
        assert src.nodata == NODATA
        assert src.transform.c % 1000 == 0
        arr = src.read(1)
        assert set(np.unique(arr)) <= {0, 1, 2, NODATA}
        assert (arr != NODATA).sum() == len(cells)
        assert src.colormap(1)[0][3] == 255
    assert (run_dir / "map" / "clusters_k3.png").stat().st_size > 1000

    prov = json.loads((run_dir / "provenance.json").read_text())
    assert prov["year"] == 2024 and "commit" in prov["git"]
    assert prov["tessera_embedding_version"]["dataset_variant"] == "synthetic"
    assert (run_dir / "config.resolved.yaml").exists()


def test_tile_sample_is_reproducible_and_order_independent() -> None:
    from tessera_nordic.pipeline.aggregate import sample_units
    from tessera_nordic.sources.base import WorkUnit

    units = [WorkUnit(f"2024_{i:03d}") for i in range(200)]
    a = sample_units(units, 0.1, seed=3)
    b = sample_units(list(reversed(units)), 0.1, seed=3)
    assert len(a) == 20 and [u.unit_id for u in a] == [u.unit_id for u in b]
    assert [u.unit_id for u in sample_units(units, 0.1, seed=4)] != [u.unit_id for u in a]
