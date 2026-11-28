"""Two-sample statistics on synthetic Gaussian data."""

from __future__ import annotations

import numpy as np
import pytest

from tessera_nordic import twosample as ts


@pytest.fixture
def samples() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    a = rng.normal(size=(600, 4))
    same = rng.normal(size=(600, 4))
    shifted = rng.normal(size=(600, 4)) + np.array([1.5, 0, 0, 0])
    return a, same, shifted


def test_distances_order(samples) -> None:
    a, same, shifted = samples
    rng = np.random.default_rng(1)
    bw = ts.median_bandwidth(np.vstack([a, shifted]), rng)
    assert abs(ts.mmd2(a, same, bw)) < ts.mmd2(a, shifted, bw)
    d = ts.random_directions(4, 50, rng)
    assert np.allclose(np.linalg.norm(d, axis=0), 1)
    assert ts.sliced_wasserstein(a @ d, same @ d) < ts.sliced_wasserstein(a @ d, shifted @ d)


def test_c2st(samples) -> None:
    a, same, shifted = samples
    cv = np.arange(1200) % 10
    t = np.r_[np.zeros(600, int), np.ones(600, int)]
    for model in ("logistic", "hgb"):
        auc_same, _ = ts.auc_and_tv(t, ts.c2st(np.vstack([a, same]), t, cv, model, 5, 0))
        auc_shift, tv = ts.auc_and_tv(t, ts.c2st(np.vstack([a, shifted]), t, cv, model, 5, 0))
        assert abs(auc_same - 0.5) < 0.08
        assert auc_shift > 0.8 and tv > 0.4


def test_coverage(samples) -> None:
    a, same, _ = samples
    k = 5
    assert ts.coverage(a, same, k).mean() == pytest.approx(1 - 0.5**k, abs=0.04)
    far = same + 10
    assert ts.coverage(a, far, k).mean() == 0


def test_halfplane_split() -> None:
    rng = np.random.default_rng(0)
    x, y = rng.uniform(size=1000), rng.uniform(size=1000)
    south = ts.halfplane_split(x, y, 90, 0.3)
    assert south.mean() == pytest.approx(0.3, abs=0.01)
    assert y[south].max() <= y[~south].min()


def test_coverage_ignores_own_block() -> None:
    rng = np.random.default_rng(0)
    # 40 blocks of 25 near-duplicate points: without blocks the radius is tiny.
    centres = rng.normal(size=(40, 3))
    ref = np.repeat(centres, 25, axis=0) + rng.normal(scale=1e-3, size=(1000, 3))
    blocks = np.repeat(np.arange(40), 25)
    other_c = rng.normal(size=(40, 3))
    other = np.repeat(other_c, 25, axis=0) + rng.normal(scale=1e-3, size=(1000, 3))
    assert ts.coverage(ref, other, 5).mean() < 0.1
    # Clumps of 25: k=5 means "nearest other clump", so about half are covered.
    assert ts.coverage(ref, other, 5, blocks).mean() > 0.3
    r = ts.knn_radius(ref, 5, blocks)
    assert np.isfinite(r).all() and (r > 0.01).all()
