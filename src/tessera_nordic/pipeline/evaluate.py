"""Stage 4 - evaluate: internal validity metrics and seed stability per k."""

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
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from tessera_nordic.config import RunConfig
from tessera_nordic.methods import get_method
from tessera_nordic.pipeline.cluster import (
    CLUSTER_DIR,
    LABELS_FILE,
    PREPROCESS_FILE,
    feature_matrix,
    label_col,
    load_cells,
    model_file,
)

log = logging.getLogger(__name__)

EVAL_DIR = "evaluate"
METRICS_FILE = "metrics.csv"


def stability(
    Z: np.ndarray, k: int, seeds: list[int], method_name: str, params: dict[str, Any]
) -> tuple[float, float, float]:
    """Pairwise adjusted Rand index between fits with different seeds.

    Returns:
        ``(mean, std, min)`` ARI over all seed pairs.
    """
    method = get_method(method_name, params)
    runs = [method.predict(method.fit(Z, k, s), Z) for s in seeds]
    aris = [adjusted_rand_score(a, b) for a, b in itertools.combinations(runs, 2)]
    return float(np.mean(aris)), float(np.std(aris)), float(np.min(aris))


def compute_metrics(
    Z: np.ndarray, labels: dict[int, np.ndarray], config: RunConfig, models: dict[int, Any]
) -> pd.DataFrame:
    """Silhouette (sampled), Davies-Bouldin, Calinski-Harabasz, inertia and stability per k."""
    cfg = config.cluster
    rng = np.random.default_rng(config.seed)
    n = Z.shape[0]
    sample = rng.choice(n, size=min(n, config.evaluate.silhouette_sample), replace=False)
    method = get_method(cfg.method, cfg.params)
    rows = []
    for k, lab in sorted(labels.items()):
        n_labels = len(np.unique(lab))
        row: dict[str, Any] = {"k": k, "n_cells": n, "n_nonempty_clusters": n_labels}
        if 2 <= n_labels < n:
            sample_labels = lab[sample]
            ok = 2 <= len(np.unique(sample_labels)) < len(sample)
            row["silhouette"] = (
                float(silhouette_score(Z[sample], sample_labels, random_state=config.seed))
                if ok
                else np.nan
            )
            row["davies_bouldin"] = float(davies_bouldin_score(Z, lab))
            row["calinski_harabasz"] = float(calinski_harabasz_score(Z, lab))
        row["inertia"] = method.inertia(models[k], Z) if k in models else np.nan
        mean, std, mn = stability(Z, k, config.evaluate.stability_seeds, cfg.method, cfg.params)
        row.update({"stability_ari_mean": mean, "stability_ari_std": std, "stability_ari_min": mn})
        sizes = np.bincount(lab, minlength=k) / n
        row.update(
            {
                "smallest_cluster_frac": float(sizes.min()),
                "largest_cluster_frac": float(sizes.max()),
            }
        )
        rows.append(row)
        log.info(
            "k=%d silhouette=%.3f DB=%.3f CH=%.1f ARI=%.3f",
            k,
            row.get("silhouette", np.nan),
            row.get("davies_bouldin", np.nan),
            row.get("calinski_harabasz", np.nan),
            mean,
        )
    return pd.DataFrame(rows)


def plot_metrics(metrics: pd.DataFrame, path: Path, dpi: int) -> None:
    """One small panel per metric, against k."""
    panels = [
        ("silhouette", "Silhouette (higher is better)"),
        ("davies_bouldin", "Davies-Bouldin (lower is better)"),
        ("calinski_harabasz", "Calinski-Harabasz (higher is better)"),
        ("inertia", "Inertia (elbow)"),
        ("stability_ari_mean", "Seed stability, mean ARI"),
    ]
    fig, axes = plt.subplots(
        1, len(panels), figsize=(4 * len(panels), 3.2), constrained_layout=True
    )
    for ax, (col, title) in zip(axes, panels, strict=True):
        if col in metrics:
            ax.plot(metrics["k"], metrics[col], marker="o")
            if col == "stability_ari_mean":
                ax.fill_between(
                    metrics["k"],
                    metrics[col] - metrics["stability_ari_std"],
                    metrics[col] + metrics["stability_ari_std"],
                    alpha=0.2,
                )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("k")
        ax.set_xticks(metrics["k"])
        ax.grid(alpha=0.3)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_explained_variance(ev: pd.DataFrame, path: Path, dpi: int) -> None:
    """Per-component and cumulative PCA explained variance."""
    fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    ax.bar(ev["component"], ev["explained_variance_ratio"], label="per component")
    ax.plot(ev["component"], ev["cumulative"], marker="o", color="C1", label="cumulative")
    ax.set_xlabel("PCA component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def run_evaluate(config: RunConfig, run_dir: Path) -> Path:
    """Write ``evaluate/metrics.csv`` and plots for every fitted k.

    Returns:
        Path of the metrics CSV.
    """
    cdir = run_dir / CLUSTER_DIR
    out = run_dir / EVAL_DIR
    out.mkdir(exist_ok=True)
    df = load_cells(run_dir)
    X, _ = feature_matrix(df, config.cluster.feature_prefixes)
    pre = joblib.load(cdir / PREPROCESS_FILE)
    Z = pre.transform(X)
    lab_df = pd.read_parquet(cdir / LABELS_FILE)
    if not np.array_equal(lab_df["cell_id"].to_numpy(), df["cell_id"].to_numpy()):
        raise ValueError("labels and cells are out of sync; rerun cluster")
    labels = {
        k: lab_df[label_col(k)].to_numpy() for k in config.cluster.k if label_col(k) in lab_df
    }
    models = {k: joblib.load(cdir / model_file(k)) for k in labels}
    metrics = compute_metrics(Z, labels, config, models)
    path = out / METRICS_FILE
    metrics.to_csv(path, index=False)
    plot_metrics(metrics, out / "metrics_vs_k.png", config.map.dpi)
    ev = pd.read_csv(cdir / "pca_explained_variance.csv")
    plot_explained_variance(ev, out / "pca_explained_variance.png", config.map.dpi)
    log.info("Wrote %s", path)
    return path
