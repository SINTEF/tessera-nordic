"""Stage 3 - cluster: optional L2 normalisation, PCA, then one model per k."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from tessera_nordic.config import ClusterConfig, RunConfig
from tessera_nordic.methods import get_method
from tessera_nordic.pipeline.merge import CELLS_FILE
from tessera_nordic.stats import CELL_ID

log = logging.getLogger(__name__)

CLUSTER_DIR = "cluster"
LABELS_FILE = "labels.parquet"
PREPROCESS_FILE = "preprocess.joblib"


def label_col(k: int) -> str:
    """Column of the labels for ``k`` clusters."""
    return f"label_k{k}"


def model_file(k: int) -> str:
    """File name of the fitted model for ``k`` clusters."""
    return f"model_k{k}.joblib"


def load_cells(run_dir: Path) -> pd.DataFrame:
    """Read the merged cell table (without geometry) from a run directory."""
    path = run_dir / CELLS_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run merge first")
    df = pd.read_parquet(path)
    return df.drop(columns=["geometry"], errors="ignore")


def feature_matrix(df: pd.DataFrame, prefixes: list[str]) -> tuple[np.ndarray, list[str]]:
    """Select feature columns by prefix.

    Returns:
        ``(X, columns)`` with ``X`` float64.
    """
    cols = [c for c in df.columns if any(c.startswith(p) for p in prefixes)]
    if not cols:
        raise ValueError(f"no columns match prefixes {prefixes}")
    return df[cols].to_numpy(np.float64), cols


def build_preprocessor(cfg: ClusterConfig, n_samples: int, n_features: int, seed: int) -> Pipeline:
    """L2 normalisation (optional) followed by PCA.

    ``n_components`` is capped at ``min(n_samples, n_features)``.
    """
    n_comp = min(cfg.pca.n_components, n_samples, n_features)
    if n_comp < cfg.pca.n_components:
        log.warning("PCA n_components reduced from %d to %d", cfg.pca.n_components, n_comp)
    steps: list[tuple[str, object]] = []
    if cfg.l2_normalize:
        steps.append(("l2", Normalizer(norm="l2")))
    steps.append(("pca", PCA(n_components=n_comp, whiten=cfg.pca.whiten, random_state=seed)))
    return Pipeline(steps)


def run_cluster(config: RunConfig, run_dir: Path) -> Path:
    """Fit the preprocessor and one clustering model per k; save models and labels.

    Outputs in ``run_dir/cluster/``: ``preprocess.joblib``, ``model_k<k>.joblib``,
    ``labels.parquet`` (``cell_id`` + ``label_k<k>``) and ``pca_explained_variance.csv``.

    Returns:
        The cluster output directory.
    """
    cfg = config.cluster
    out = run_dir / CLUSTER_DIR
    out.mkdir(exist_ok=True)
    df = load_cells(run_dir)
    X, cols = feature_matrix(df, cfg.feature_prefixes)
    log.info("Clustering %d cells x %d features", X.shape[0], X.shape[1])

    pre = build_preprocessor(cfg, X.shape[0], X.shape[1], config.seed)
    Z = pre.fit_transform(X)
    joblib.dump(pre, out / PREPROCESS_FILE)
    pca: PCA = pre.named_steps["pca"]
    ev = pd.DataFrame(
        {
            "component": np.arange(1, pca.n_components_ + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    ev.to_csv(out / "pca_explained_variance.csv", index=False)
    log.info("PCA: %d components explain %.1f%% of variance", len(ev), 100 * ev.cumulative.iloc[-1])

    method = get_method(cfg.method, cfg.params)
    labels = pd.DataFrame({CELL_ID: df[CELL_ID].to_numpy()})
    for k in cfg.k:
        if k > len(df):
            log.warning("Skipping k=%d: only %d cells", k, len(df))
            continue
        model = method.fit(Z, k, config.seed)
        labels[label_col(k)] = method.predict(model, Z)
        joblib.dump(model, out / model_file(k))
        sizes = np.bincount(labels[label_col(k)], minlength=k)
        log.info("k=%d: cluster sizes min %d, max %d", k, sizes.min(), sizes.max())
    labels.to_parquet(out / LABELS_FILE, index=False)
    (out / "feature_columns.txt").write_text("\n".join(cols) + "\n", encoding="utf-8")
    return out
