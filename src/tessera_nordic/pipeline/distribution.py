"""Embedding-space comparison of groups, with a reference scale from pseudo-groups.

Part of stage 6 (compare). Where :mod:`tessera_nordic.pipeline.compare` compares the
groups' shares of the shared clusters, this module compares the groups' distributions
in the (L2-normalised, PCA) embedding space directly, so the result does not depend on
``k`` or on one clustering. For every pair of groups it reports:

- ``c2st_auc`` / ``tv_lower_bound``: a classifier two-sample test. A classifier learns
  to tell the groups apart, evaluated on held-out spatial blocks. AUC 0.5 means
  indistinguishable; ``tv_lower_bound`` (|2 * balanced accuracy - 1|) is a lower bound
  on the share of the two distributions that does not overlap.
- ``energy``, ``mmd2`` and ``swd``: energy distance, kernel MMD (squared, RBF kernel
  with one bandwidth for all comparisons) and sliced Wasserstein-1 distance (in
  embedding units).
- ``uncovered_a`` / ``uncovered_b``: share of a group's cells with no counterpart in the
  other group (k-NN coverage). This separates "same landscape types in different
  proportions" (low) from "types found in only one group" (high).
- ``jsd_k<k>``: the cluster-share JSD, for side-by-side comparison.

Every statistic is "higher = more different". Two reference scales make the numbers
interpretable:

- *within block halves*: one group split into two random halves of its spatial
  blocks, i.e. the same distribution sampled in different places;
- *pseudo-groups*: the pooled area cut by straight lines at many angles into two
  contiguous regions with the same size ratio as the real pair ("pooled"), and each
  group cut by lines into two halves ("within <g>"). The percentile of the real pair
  among these says whether the real border separates the landscape more, or less,
  than an arbitrary line would.
"""

from __future__ import annotations

import itertools
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tessera_nordic import twosample as ts
from tessera_nordic.config import CompareConfig

log = logging.getLogger(__name__)

BASE_METRICS = [
    "c2st_auc",
    "tv_lower_bound",
    "energy",
    "mmd2",
    "swd",
    "uncovered_a",
    "uncovered_b",
]


@dataclass
class Space:
    """Cells in embedding space plus everything the statistics need."""

    z: np.ndarray
    x: np.ndarray
    y: np.ndarray
    blocks: np.ndarray
    labels: dict[int, np.ndarray]
    cfg: CompareConfig
    seed: int
    bandwidth: float
    proj: np.ndarray

    @classmethod
    def build(
        cls,
        z: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        blocks: np.ndarray,
        labels: dict[int, np.ndarray],
        cfg: CompareConfig,
        seed: int,
    ) -> Space:
        rng = np.random.default_rng(seed)
        bandwidth = ts.median_bandwidth(z, rng)
        proj = z @ ts.random_directions(z.shape[1], cfg.n_projections, rng)
        return cls(z, x, y, blocks, labels, cfg, seed, bandwidth, proj)

    def metric_names(self) -> list[str]:
        return BASE_METRICS + [f"jsd_k{k}" for k in sorted(self.labels)]


def _sample(idx: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return idx if len(idx) <= n else rng.choice(idx, size=n, replace=False)


def pair_metrics(
    sp: Space, a: np.ndarray, b: np.ndarray, rng: np.random.Generator
) -> tuple[dict[str, float], dict[str, Any]]:
    """All statistics for cells ``a`` vs cells ``b`` (boolean masks over ``sp``).

    Returns:
        ``(metrics, detail)``; ``detail`` holds per-cell results (out-of-fold
        probabilities and coverage flags) for maps and tables.
    """
    cfg = sp.cfg
    ia, ib = np.flatnonzero(a), np.flatnonzero(b)
    n_a, n_b = len(ia), len(ib)
    # Classifier and sliced Wasserstein use at most `max_cells_per_group` cells per group.
    ca = np.sort(_sample(ia, cfg.max_cells_per_group, rng))
    cb = np.sort(_sample(ib, cfg.max_cells_per_group, rng))
    idx = np.concatenate([ca, cb])
    target = np.r_[np.zeros(len(ca), int), np.ones(len(cb), int)]
    proba = ts.c2st(sp.z[idx], target, sp.blocks[idx], cfg.c2st_model, cfg.n_folds, sp.seed)
    auc, tv = ts.auc_and_tv(target, proba)

    m = min(len(ia), len(ib), cfg.coverage_sample)
    ra, rb = _sample(ia, m, rng), _sample(ib, m, rng)
    cov_a = ts.coverage(sp.z[ra], sp.z[rb], cfg.coverage_k, sp.blocks[ra])
    cov_b = ts.coverage(sp.z[rb], sp.z[ra], cfg.coverage_k, sp.blocks[rb])

    out: dict[str, float] = {
        "n_a": float(n_a),
        "n_b": float(n_b),
        "c2st_auc": auc,
        "tv_lower_bound": tv,
        "energy": ts.energy_distance(
            sp.z[_sample(ia, cfg.energy_sample, rng)], sp.z[_sample(ib, cfg.energy_sample, rng)]
        ),
        "mmd2": ts.mmd2(
            sp.z[_sample(ia, cfg.mmd_sample, rng)],
            sp.z[_sample(ib, cfg.mmd_sample, rng)],
            sp.bandwidth,
        ),
        "swd": ts.sliced_wasserstein(sp.proj[ca], sp.proj[cb]),
        "uncovered_a": 1 - float(cov_a.mean()),
        "uncovered_b": 1 - float(cov_b.mean()),
    }
    for k, lab in sorted(sp.labels.items()):
        out[f"jsd_k{k}"] = ts.jsd(
            np.bincount(lab[ia], minlength=k) + 0.0, np.bincount(lab[ib], minlength=k) + 0.0
        )
    detail = {
        "idx": idx,
        "target": target,
        "proba": proba,
        "cov_idx_a": ra,
        "cov_a": cov_a,
        "cov_idx_b": rb,
        "cov_b": cov_b,
    }
    return out, detail


def bootstrap_ci(
    sp: Space, a: np.ndarray, b: np.ndarray, detail: dict[str, Any], rng: np.random.Generator
) -> dict[str, float]:
    """95% block-bootstrap intervals for the C2ST AUC (fixed out-of-fold scores) and SWD."""
    idx, target, proba = detail["idx"], detail["target"], detail["proba"]
    _, inv = np.unique(sp.blocks[idx], return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(inv.max() + 1)]
    aucs, swds = [], []
    for _ in range(sp.cfg.n_bootstrap):
        pick = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        t, p = target[pick], proba[pick]
        if len(np.unique(t)) < 2:
            continue
        aucs.append(ts.auc_and_tv(t, p)[0])
        cells = idx[pick]
        swds.append(ts.sliced_wasserstein(sp.proj[cells[t == 0]], sp.proj[cells[t == 1]]))
    lo, hi = 2.5, 97.5
    return {
        "c2st_auc_ci_low": float(np.nanpercentile(aucs, lo)),
        "c2st_auc_ci_high": float(np.nanpercentile(aucs, hi)),
        "swd_ci_low": float(np.percentile(swds, lo)),
        "swd_ci_high": float(np.percentile(swds, hi)),
    }


def block_halves(
    sp: Space, mask: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Split ``mask`` into two random halves of its spatial blocks."""
    ub = np.unique(sp.blocks[mask])
    half = rng.permutation(ub)[: len(ub) // 2]
    in_half = np.isin(sp.blocks, half)
    return mask & in_half, mask & ~in_half


def _angles(n: int, full_turn: bool) -> np.ndarray:
    return np.linspace(0, 360 if full_turn else 180, n, endpoint=False)


def reference_splits(
    sp: Space, groups: np.ndarray, a: str, b: str, rng: np.random.Generator
) -> list[dict[str, Any]]:
    """Statistics for straight-line pseudo-group splits (see module docstring)."""
    cfg = sp.cfg
    rows: list[dict[str, Any]] = []
    if cfg.region_angles == 0:
        return rows
    pooled = (groups == a) | (groups == b)
    share_a = float((groups == a).sum() / pooled.sum())
    started = time.monotonic()
    families = [("pooled", pooled, share_a, True)] + [
        (f"within {g}", groups == g, 0.5, False) for g in (a, b)
    ]
    for family, mask, share, full_turn in families:
        ii = np.flatnonzero(mask)
        for angle in _angles(cfg.region_angles, full_turn):
            side = ts.halfplane_split(sp.x[ii], sp.y[ii], float(angle), share)
            pa = np.zeros(len(groups), bool)
            pb = np.zeros(len(groups), bool)
            pa[ii[side]] = True
            pb[ii[~side]] = True
            if min(pa.sum(), pb.sum()) < 20:
                continue
            met, _ = pair_metrics(sp, pa, pb, rng)
            rows.append({"pair": f"{a} vs {b}", "family": family, "angle_deg": angle, **met})
        log.info(
            "Reference %s vs %s, family %r: %d splits done (%.0f s so far)",
            a,
            b,
            family,
            cfg.region_angles,
            time.monotonic() - started,
        )
    return rows


def percentiles(real: pd.DataFrame, ref: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Where each real pair falls among its pseudo-group splits, per family and metric."""
    rows = []
    for _, r in real.iterrows():
        for family, d in ref[ref.pair == r["pair"]].groupby("family", sort=False):
            for mname in metrics:
                vals = d[mname].dropna().to_numpy()
                if not len(vals):
                    continue
                rows.append(
                    {
                        "pair": r["pair"],
                        "family": family,
                        "metric": mname,
                        "value": r[mname],
                        "ref_min": float(vals.min()),
                        "ref_median": float(np.median(vals)),
                        "ref_max": float(vals.max()),
                        "percentile": float(100 * np.mean(vals < r[mname])),
                        "n_splits": len(vals),
                    }
                )
    return pd.DataFrame(rows)


def plot_reference(
    real: pd.DataFrame,
    within: pd.DataFrame,
    ref: pd.DataFrame,
    metrics: list[str],
    path: Path,
    dpi: int,
) -> None:
    """One panel per statistic: pseudo-group splits per family vs the real pair."""
    pair = real.iloc[0]
    d_ref = ref[ref.pair == pair["pair"]]
    families = list(dict.fromkeys(d_ref.family))
    ncol = 4
    nrow = int(np.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(3.4 * ncol, 2.9 * nrow), constrained_layout=True, squeeze=False
    )
    rng = np.random.default_rng(0)
    for ax, mname in zip(axes.flat, metrics, strict=False):
        for i, fam in enumerate(families):
            v = d_ref.loc[d_ref.family == fam, mname]
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=10, alpha=0.6, color="C0")
        ax.axhline(pair[mname], color="C3", lw=2, label=str(pair["pair"]))
        w = within[mname].median() if len(within) else np.nan
        ax.axhline(w, color="0.4", ls="--", lw=1, label="within (block halves)")
        ax.set_xticks(range(len(families)), families, fontsize=7)
        ax.set_title(mname, fontsize=9)
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(metrics) :]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(
        "Real pair (red) vs straight-line pseudo-group splits (dots); higher = more different",
        fontsize=10,
    )
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_maps(
    sp: Space, a: str, b: str, detail: dict[str, Any], path_c2st: Path, path_cov: Path, dpi: int
) -> None:
    """Map of the out-of-fold P(b), and of cells without a counterpart in the other group."""
    idx, proba = detail["idx"], detail["proba"]
    fig, ax = plt.subplots(figsize=(6, 7), constrained_layout=True)
    sc = ax.scatter(
        sp.x[idx] / 1000, sp.y[idx] / 1000, c=proba, s=1, cmap="coolwarm", vmin=0, vmax=1
    )
    plt.colorbar(sc, ax=ax, shrink=0.6, label=f"out-of-fold P({b})")
    ax.set_title(f"Classifier two-sample test: how {b}-like each cell looks", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlabel("km (grid CRS)")
    fig.savefig(path_c2st, dpi=dpi)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 7), constrained_layout=True)
    for ax, (g, other, ci, cov) in zip(
        axes,
        [
            (a, b, detail["cov_idx_a"], detail["cov_a"]),
            (b, a, detail["cov_idx_b"], detail["cov_b"]),
        ],
        strict=True,
    ):
        ax.scatter(sp.x[ci] / 1000, sp.y[ci] / 1000, s=1, color="0.8")
        unc = ci[~cov]
        ax.scatter(sp.x[unc] / 1000, sp.y[unc] / 1000, s=2, color="C3")
        ax.set_title(
            f"{g} cells without a counterpart in {other}: {100 * (1 - cov.mean()):.1f}%",
            fontsize=10,
        )
        ax.set_aspect("equal")
    fig.savefig(path_cov, dpi=dpi)
    plt.close(fig)


def coverage_by_cluster(sp: Space, a: str, b: str, detail: dict[str, Any]) -> pd.DataFrame:
    """Share of each group's cells without a counterpart in the other group, per cluster."""
    rows = []
    for g, ci, cov in (
        (a, detail["cov_idx_a"], detail["cov_a"]),
        (b, detail["cov_idx_b"], detail["cov_b"]),
    ):
        for k, lab in sorted(sp.labels.items()):
            d = pd.DataFrame({"cluster": lab[ci], "uncovered": ~cov})
            agg = d.groupby("cluster").uncovered.agg(["size", "mean"]).reset_index()
            for _, r in agg.iterrows():
                rows.append(
                    {
                        "k": k,
                        "group": g,
                        "cluster": int(r["cluster"]),
                        "n_cells_sampled": int(r["size"]),
                        "uncovered_share": float(r["mean"]),
                    }
                )
    return pd.DataFrame(rows)


def run_distribution(
    sp: Space, groups: np.ndarray, names: list[str], out: Path, dpi: int
) -> pd.DataFrame:
    """Write the embedding-space comparison outputs to ``out``; return the summary table.

    Files: ``distribution_summary.csv`` (real pairs and within-group block halves),
    ``reference_splits.csv``, ``reference_percentiles.csv``, ``coverage_by_cluster.csv``,
    ``cell_scores.parquet`` and the figures ``reference.png``, ``c2st_map.png``,
    ``coverage_map.png`` (figures and per-cell outputs for the first pair).
    """
    rng = np.random.default_rng(sp.seed)
    metrics = sp.metric_names()
    real_rows, within_rows, ref_rows = [], [], []
    scores = pd.DataFrame({"row": np.arange(len(groups)), "group": groups})
    cov_tables = []

    for g in names:
        reps = []
        for _ in range(sp.cfg.within_reps):
            h1, h2 = block_halves(sp, groups == g, rng)
            if min(h1.sum(), h2.sum()) < 20:
                continue
            reps.append(pair_metrics(sp, h1, h2, rng)[0])
        if reps:
            med = pd.DataFrame(reps).median().to_dict()
            within_rows.append({"comparison": f"within {g} (block halves)", "pair": "", **med})
            log.info("within %s: %s", g, _fmt(med))

    for i, (a, b) in enumerate(itertools.combinations(names, 2)):
        met, detail = pair_metrics(sp, groups == a, groups == b, rng)
        met.update(bootstrap_ci(sp, groups == a, groups == b, detail, rng))
        real_rows.append({"comparison": f"{a} vs {b}", "pair": f"{a} vs {b}", **met})
        log.info("%s vs %s: %s", a, b, _fmt(met))
        scores.loc[detail["idx"], f"p_{b}__{a}_vs_{b}"] = detail["proba"]
        scores.loc[detail["cov_idx_a"], f"covered_in_{b}"] = detail["cov_a"].astype(float)
        scores.loc[detail["cov_idx_b"], f"covered_in_{a}"] = detail["cov_b"].astype(float)
        cov_tables.append(coverage_by_cluster(sp, a, b, detail).assign(pair=f"{a} vs {b}"))
        if i == 0:
            plot_maps(sp, a, b, detail, out / "c2st_map.png", out / "coverage_map.png", dpi)
        ref_rows.extend(reference_splits(sp, groups, a, b, rng))

    real = pd.DataFrame(real_rows)
    within = pd.DataFrame(within_rows)
    summary = pd.concat([real, within], ignore_index=True)
    summary.to_csv(out / "distribution_summary.csv", index=False)
    pd.concat(cov_tables, ignore_index=True).to_csv(out / "coverage_by_cluster.csv", index=False)
    scores.drop(columns="row").to_parquet(out / "cell_scores.parquet", index=False)
    if ref_rows:
        ref = pd.DataFrame(ref_rows)
        ref.to_csv(out / "reference_splits.csv", index=False)
        pct = percentiles(real, ref, metrics)
        pct.to_csv(out / "reference_percentiles.csv", index=False)
        log.info(
            "Percentile of the real pair among pseudo-group splits:\n%s",
            pct.pivot_table(index="metric", columns="family", values="percentile", sort=False)
            .round(0)
            .to_string(),
        )
        mid_k = sorted(sp.labels)[len(sp.labels) // 2] if sp.labels else None
        shown = BASE_METRICS + ([f"jsd_k{mid_k}"] if mid_k is not None else [])
        plot_reference(real, within, ref, shown, out / "reference.png", dpi)
    return summary


def _fmt(met: Mapping[Any, Any]) -> str:
    keys = ["c2st_auc", "tv_lower_bound", "energy", "mmd2", "swd", "uncovered_a", "uncovered_b"]
    return ", ".join(f"{k} {met[k]:.3f}" for k in keys if k in met)
