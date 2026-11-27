"""Shared fixtures: isolated data/output directories and synthetic TESSERA tiles.

Synthetic tiles imitate the real ones: a UTM (EPSG:32632) pixel grid covering the UTM
envelope of a 0.1-degree box, int8 embeddings with per-pixel float32 scales, and a few
non-finite scales. Values depend only on the pixel position, so overlapping neighbour
tiles agree, as in the real Zarr store. Pixels are 100 m instead of 10 m to keep the
tests fast; nothing in the code assumes 10 m.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from affine import Affine

from tessera_nordic.area import TILE_DEG
from tessera_nordic.config import RunConfig, TesseraConfig
from tessera_nordic.grid import Grid, transformer
from tessera_nordic.sources.tessera import RawTile, TesseraSource

UTM = "EPSG:32632"
PIXEL = 100.0
N_BANDS = 8
SCALE = 0.05

# Three synthetic "landscapes" with distinct prototype embeddings.
PROTOTYPES = np.array(
    [
        [60, -40, 20, 0, 10, -10, 30, -20],
        [-50, 30, -20, 40, 0, 20, -30, 10],
        [0, 0, 60, -60, 40, -40, 0, 0],
    ],
    dtype=np.float64,
)


def field_values(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Deterministic int8 embeddings at UTM pixel centres ``(x, y)``, shape ``(..., B)``."""
    region = (np.floor(x / 4000) + np.floor(y / 6000)).astype(np.int64) % 3
    base = PROTOTYPES[region]
    band = np.arange(N_BANDS)
    noise = 10 * np.sin(x[..., None] * 0.0137 + y[..., None] * 0.0291 + band * 1.7)
    return np.clip(np.round(base + noise), -127, 127).astype(np.int8)


def synthetic_tile(lon: float, lat: float, nan_every: int = 97) -> RawTile:
    """A tile covering the UTM envelope of the 0.1-degree box centred at ``(lon, lat)``."""
    h = TILE_DEG / 2
    lons = np.array([lon - h, lon + h, lon - h, lon + h, lon, lon])
    lats = np.array([lat - h, lat - h, lat + h, lat + h, lat - h, lat + h])
    xs, ys = transformer("EPSG:4326", UTM).transform(lons, lats)
    x0 = math.floor(min(xs) / PIXEL) * PIXEL
    x1 = math.ceil(max(xs) / PIXEL) * PIXEL
    y0 = math.floor(min(ys) / PIXEL) * PIXEL
    y1 = math.ceil(max(ys) / PIXEL) * PIXEL
    width = int((x1 - x0) / PIXEL)
    height = int((y1 - y0) / PIXEL)
    transform = Affine(PIXEL, 0, x0, 0, -PIXEL, y1)
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    px = x0 + (cols + 0.5) * PIXEL
    py = y1 - (rows + 0.5) * PIXEL
    q = field_values(px, py)
    scales = np.full((height, width), SCALE, dtype=np.float32)
    flat = scales.reshape(-1)
    # Non-finite scales mark gaps, chosen by position so neighbours agree.
    gap = ((np.floor(px / PIXEL) * 31 + np.floor(py / PIXEL) * 17) % nan_every == 0).reshape(-1)
    flat[gap] = np.nan
    return RawTile(q, scales, transform, UTM)


class FakeReader:
    """In-memory tile reader with optional failure injection."""

    def __init__(self, fail_times: dict[tuple[float, float], int] | None = None) -> None:
        self.fail_times = dict(fail_times or {})
        self.reads: list[tuple[float, float]] = []
        self.cleaned: list[tuple[float, float]] = []

    def available(self, tiles: list[tuple[float, float]], year: int) -> list[tuple[float, float]]:
        return tiles

    def read(self, lon: float, lat: float, year: int) -> RawTile | None:
        self.reads.append((lon, lat))
        left = self.fail_times.get((lon, lat), 0)
        if left:
            self.fail_times[(lon, lat)] = left - 1
            raise ConnectionError(f"simulated network failure for {lon},{lat}")
        return synthetic_tile(lon, lat)

    def cleanup(self, lon: float, lat: float, year: int) -> None:
        self.cleaned.append((lon, lat))

    def provenance(self) -> dict[str, Any]:
        return {"backend": "fake", "dataset_version": "v1.1", "dataset_variant": "synthetic"}


def make_source(config: RunConfig, reader: FakeReader) -> TesseraSource:
    """A TESSERA source backed by synthetic tiles with :data:`N_BANDS` features."""
    grid = Grid(config.grid.crs, config.grid.cell_size)
    tcfg = TesseraConfig(**{**config.sources.tessera.model_dump(), "dataset_variant": "synthetic"})
    source = TesseraSource(tcfg, config.year, grid, reader)
    source.n_features = N_BANDS
    return source


@pytest.fixture
def env_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Point TN_DATA_DIR, TN_OUTPUT_DIR and TESSERA_DATA_DIR at a temporary directory."""
    dirs = {
        "TN_DATA_DIR": tmp_path / "data",
        "TN_OUTPUT_DIR": tmp_path / "outputs",
        "TESSERA_DATA_DIR": tmp_path / "tessera",
    }
    for name, path in dirs.items():
        monkeypatch.setenv(name, str(path))
    return dirs


@pytest.fixture
def small_config() -> RunConfig:
    """A 2 x 2 tile area near Trondheim with settings sized for synthetic 100 m pixels."""
    return RunConfig.model_validate(
        {
            "run_name": "test",
            "seed": 7,
            "year": 2024,
            "area": {"name": "test-area", "bbox": [10.3, 63.3, 10.5, 63.5]},
            "grid": {"crs": "EPSG:3035", "cell_size": 1000},
            "aggregate": {"workers": 1, "max_retries": 2, "retry_backoff_s": 0},
            "merge": {"min_pixel_count": 50},
            "cluster": {"k": [2, 3], "pca": {"n_components": 4}, "params": {"n_init": 2}},
            "evaluate": {"silhouette_sample": 500, "stability_seeds": [0, 1, 2]},
            "map": {"dpi": 60},
        }
    )
