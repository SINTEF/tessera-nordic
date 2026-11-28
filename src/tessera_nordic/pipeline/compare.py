"""Stage 6 - compare: how are groups of cells (e.g. Norway vs Sweden) distributed?

All cells share one clustering (stage 3). For each k this stage compares the groups'
distributions over those clusters:

- the share of each group's cells in every cluster, with block-bootstrap 95% intervals;
- Jensen-Shannon divergence (JSD, base 2: 0 = identical, 1 = disjoint) and histogram
  overlap (sum of min shares: 1 = identical) between each pair of groups;
- a *within-group baseline*: the same JSD between two random halves of one group's
  spatial blocks, i.e. how different two parts of the same country already look, and,
  at the same sample size, the JSD between a random half of each group
  (``jsd_between_halves``). Finite-sample JSD is biased upwards, so compare these two
  with each other rather than with the full-sample ``jsd``;
- which clusters are shared and which are specific to one group.

Independently of the clustering, :mod:`tessera_nordic.pipeline.distribution` compares
the groups directly in embedding space (classifier two-sample test, energy distance,
MMD, sliced Wasserstein, k-NN coverage) and ranks the real pair among straight-line
pseudo-group splits, which gives the numbers a reference scale.

Cells are spatially autocorrelated, so all resampling uses whole spatial blocks
(``compare.block_size``) rather than single cells.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely

from tessera_nordic.area import load_group_geometries
from tessera_nordic.config import CompareConfig, RunConfig
from tessera_nordic.grid import transformer
from tessera_nordic.pipeline.cluster import (
    CLUSTER_DIR,
    LABELS_FILE,
    PREPROCESS_FILE,
    feature_matrix,
    label_col,
    load_cells,
)
from tessera_nordic.pipeline.distribution import Space, run_distribution
from tessera_nordic.twosample import energy_distance, jsd, overlap

__all__ = ["energy_distance", "jsd", "overlap", "run_compare"]

log = logging.getLogger(__name__)

COMPARE_DIR = "compare"


def assign_groups(x: np.ndarray, y: np.ndarray, crs: str, cfg: CompareConfig) -> np.ndarray:
    """Group name of the polygon containing each cell centre (``""`` if none)."""
    geoms = load_group_geometries(cfg.groups)
    lon, lat = transformer(crs, "EPSG:4326").transform(x, y)
    out = np.full(len(x), "", dtype=object)
    for name, geom in geoms.items():
        inside = shapely.contains_xy(geom, lon, lat)
        out[inside & (out == "")] = name
    return out


def block_ids(x: np.ndarray, y: np.ndarray, size: int) -> np.ndarray:
    """Integer id of the spatial block of each cell."""
    bx = np.floor(x / size).astype(np.int64)
    by = np.floor(y / size).astype(np.int64)
    return bx * 1_000_003 + by


class _GroupBlocks:
    """Per-group block x cluster count matrices, for fast block resampling."""

    def __init__(self, labels: np.ndarray, groups: np.ndarray, blocks: np.ndarray, k: int):
        self.k = k
        self.counts: dict[str, np.ndarray] = {}
        for g in np.unique(groups):
            sel = groups == g
            _, inv = np.unique(blocks[sel], return_inverse=True)
            mat = np.zeros((inv.max() + 1, k), dtype=np.int64)
            np.add.at(mat, (inv, labels[sel]), 1)
            self.counts[str(g)] = mat

    def total(self, g: str) -> np.ndarray:
        return self.counts[g].sum(axis=0)

    def resample(self, g: str, rng: np.random.Generator) -> np.ndarray:
        mat = self.counts[g]
        return mat[rng.integers(0, len(mat), len(mat))].sum(axis=0)

    def split_halves(self, g: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        mat = self.counts[g]
        perm = rng.permutation(len(mat))
        half = len(mat) // 2
        return mat[perm[:half]].sum(axis=0), mat[perm[half:]].sum(axis=0)


def _ordered(cfg: CompareConfig, present: Any) -> list[str]:
    """Groups in the order given in the config, keeping only those that have cells."""
    return [str(v) for v in (cfg.groups.values or []) if v in present]


def _ci(values: list[float]) -> tuple[float, float]:
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def compare_clusters(
    labels: np.ndarray,
    groups: np.ndarray,
    blocks: np.ndarray,
    k: int,
    cfg: CompareConfig,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Cluster shares per group and pairwise divergence statistics for one k.

    Returns:
        ``(shares, pair_rows)``: a long table of shares with 95% intervals, and one
        summary row per pair of groups.
    """
    rng = np.random.default_rng(seed)
    gb = _GroupBlocks(labels, groups, blocks, k)
    names = _ordered(cfg, gb.counts)
    boot = {g: np.array([gb.resample(g, rng) for _ in range(cfg.n_bootstrap)]) for g in names}

    share_rows = []
    for g in names:
        total = gb.total(g)
        n = total.sum()
        bshares = boot[g] / boot[g].sum(axis=1, keepdims=True)
        lo, hi = np.percentile(bshares, [2.5, 97.5], axis=0)
        for c in range(k):
            share_rows.append(
                {
                    "k": k,
                    "cluster": c,
                    "group": g,
                    "n_cells": int(total[c]),
                    "share": float(total[c] / n),
                    "share_ci_low": float(lo[c]),
                    "share_ci_high": float(hi[c]),
                }
            )
    shares = pd.DataFrame(share_rows)

    within: dict[str, list[float]] = {
        g: [jsd(*gb.split_halves(g, rng)) for _ in range(cfg.n_bootstrap)] for g in names
    }

    pair_rows = []
    for a, b in itertools.combinations(names, 2):
        pa, pb = gb.total(a), gb.total(b)
        j_boot = [jsd(boot[a][i], boot[b][i]) for i in range(cfg.n_bootstrap)]
        o_boot = [overlap(boot[a][i], boot[b][i]) for i in range(cfg.n_bootstrap)]
        # Same sample size as the within-group baseline: half of A's blocks vs half of B's.
        between = [
            jsd(gb.split_halves(a, rng)[0], gb.split_halves(b, rng)[0])
            for _ in range(cfg.n_bootstrap)
        ]
        sa, sb = pa / pa.sum(), pb / pb.sum()
        present_a, present_b = sa >= cfg.min_share, sb >= cfg.min_share
        row: dict[str, Any] = {
            "k": k,
            "group_a": a,
            "group_b": b,
            "n_cells_a": int(pa.sum()),
            "n_cells_b": int(pb.sum()),
            "n_blocks_a": len(gb.counts[a]),
            "n_blocks_b": len(gb.counts[b]),
            "jsd": jsd(pa, pb),
            "jsd_ci_low": _ci(j_boot)[0],
            "jsd_ci_high": _ci(j_boot)[1],
            "overlap": overlap(pa, pb),
            "overlap_ci_low": _ci(o_boot)[0],
            "overlap_ci_high": _ci(o_boot)[1],
            "jsd_between_halves_median": float(np.median(between)),
            "jsd_between_halves_p5": float(np.percentile(between, 5)),
            f"jsd_within_{a}_median": float(np.median(within[a])),
            f"jsd_within_{a}_p95": float(np.percentile(within[a], 95)),
            f"jsd_within_{b}_median": float(np.median(within[b])),
            f"jsd_within_{b}_p95": float(np.percentile(within[b], 95)),
            "n_clusters_shared": int(np.sum(present_a & present_b)),
            f"clusters_only_{a}": " ".join(str(c) for c in np.flatnonzero(present_a & ~present_b)),
            f"clusters_only_{b}": " ".join(str(c) for c in np.flatnonzero(present_b & ~present_a)),
        }
        pair_rows.append(row)
        log.info(
            "k=%d %s vs %s: JSD %.3f [%.3f, %.3f] (within %s %.3f, %s %.3f), overlap %.2f, "
            "%d shared clusters",
            k,
            a,
            b,
            row["jsd"],
            row["jsd_ci_low"],
            row["jsd_ci_high"],
            a,
            row[f"jsd_within_{a}_median"],
            b,
            row[f"jsd_within_{b}_median"],
            row["overlap"],
            row["n_clusters_shared"],
        )
    return shares, pair_rows


def plot_shares(shares: pd.DataFrame, k: int, path: Path, dpi: int) -> None:
    """Grouped bars of cluster shares per group with 95% block-bootstrap intervals."""
    d = shares[shares.k == k]
    names = sorted(d.group.unique())
    wide = d.pivot(index="cluster", columns="group", values="share")
    order = (wide[names[0]] - wide[names[1]]).sort_values().index if len(names) == 2 else wide.index
    width = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * k + 2), 3.8), constrained_layout=True)
    pos = np.arange(len(order))
    for i, g in enumerate(names):
        dg = d[d.group == g].set_index("cluster").loc[order]
        err = np.clip(np.vstack([dg.share - dg.share_ci_low, dg.share_ci_high - dg.share]), 0, None)
        ax.bar(
            pos + (i - (len(names) - 1) / 2) * width, dg.share, width, yerr=err, label=g, capsize=2
        )
    ax.set_xticks(pos, [str(c) for c in order])
    ax.set_xlabel("cluster (sorted by difference in share)")
    ax.set_ylabel("share of the group's cells")
    ax.set_title(
        f"Cluster composition by group, k={k} (95% block-bootstrap intervals)", fontsize=10
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_divergence(summary: pd.DataFrame, path: Path, dpi: int) -> None:
    """JSD between groups vs k, next to the within-group baselines."""
    fig, ax = plt.subplots(figsize=(6, 3.8), constrained_layout=True)
    for (a, b), d in summary.groupby(["group_a", "group_b"]):
        ax.errorbar(
            d.k,
            d.jsd,
            # Finite-sample JSD is biased upwards, so the bootstrap interval can sit above
            # the point estimate; clip the bars, the CSV keeps the raw interval.
            yerr=np.clip(np.vstack([d.jsd - d.jsd_ci_low, d.jsd_ci_high - d.jsd]), 0, None),
            marker="o",
            capsize=3,
            label=f"{a} vs {b}",
        )
        for g in (a, b):
            ax.plot(
                d.k, d[f"jsd_within_{g}_median"], ls="--", marker=".", label=f"within {g} (halves)"
            )
        ax.plot(d.k, d.jsd_between_halves_median, ls=":", marker="s", label=f"{a} vs {b} (halves)")
    ax.set_xlabel("k")
    ax.set_ylabel("Jensen-Shannon divergence")
    ax.set_xticks(sorted(summary.k.unique()))
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def run_compare(config: RunConfig, run_dir: Path) -> Path:
    """Write ``compare/`` outputs: shares, pairwise summary, energy distances and plots.

    Returns:
        Path of ``compare/summary.csv``.
    """
    if config.compare is None:
        raise ValueError("the config has no `compare` section")
    cfg = config.compare
    out = run_dir / COMPARE_DIR
    out.mkdir(exist_ok=True)

    df = load_cells(run_dir)
    lab = pd.read_parquet(run_dir / CLUSTER_DIR / LABELS_FILE)
    if not np.array_equal(lab["cell_id"].to_numpy(), df["cell_id"].to_numpy()):
        raise ValueError("labels and cells are out of sync; rerun cluster")
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    groups = assign_groups(x, y, config.grid.crs, cfg)
    keep = groups != ""
    log.info(
        "Groups: %s; %d cells outside every group dropped",
        pd.Series(groups[keep]).value_counts().to_dict(),
        int((~keep).sum()),
    )
    pd.DataFrame({"cell_id": df["cell_id"], "group": groups}).to_parquet(
        out / "cell_groups.parquet", index=False
    )
    if len(set(groups[keep])) < 2:
        raise ValueError("fewer than two groups contain cells")
    blocks = block_ids(x, y, cfg.block_size)

    all_shares, all_pairs = [], []
    for k in config.cluster.k:
        if label_col(k) not in lab:
            continue
        labels = lab[label_col(k)].to_numpy()
        shares, pairs = compare_clusters(
            labels[keep], groups[keep], blocks[keep], k, cfg, config.seed
        )
        all_shares.append(shares)
        all_pairs.extend(pairs)
        plot_shares(shares, k, out / f"shares_k{k}.png", config.map.dpi)
    shares_df = pd.concat(all_shares, ignore_index=True)
    shares_df.to_csv(out / "cluster_shares.csv", index=False)
    summary = pd.DataFrame(all_pairs)
    path = out / "summary.csv"
    summary.to_csv(path, index=False)
    plot_divergence(summary, out / "divergence_vs_k.png", config.map.dpi)

    X, _ = feature_matrix(df, config.cluster.feature_prefixes)
    Z = joblib.load(run_dir / CLUSTER_DIR / PREPROCESS_FILE).transform(X)
    sp = Space.build(
        Z[keep],
        x[keep],
        y[keep],
        blocks[keep],
        {k: lab[label_col(k)].to_numpy()[keep] for k in config.cluster.k if label_col(k) in lab},
        cfg,
        config.seed,
    )
    scores_groups = groups[keep].astype(str)
    run_distribution(sp, scores_groups, _ordered(cfg, set(scores_groups)), out, config.map.dpi)
    cell_scores = pd.read_parquet(out / "cell_scores.parquet")
    cell_scores.insert(0, "cell_id", df["cell_id"].to_numpy()[keep])
    cell_scores.to_parquet(out / "cell_scores.parquet", index=False)
    log.info("Wrote %s", out)
    return path
