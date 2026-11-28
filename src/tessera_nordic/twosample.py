"""Two-sample statistics for comparing groups of cells in embedding space.

All functions are pure (arrays in, numbers out) so they can be reused for the real
group pair and for the pseudo-group reference splits. "Higher = more different" holds
for every statistic returned here, except :func:`coverage`, which callers turn into an
"uncovered" share.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, pairwise_distances, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

C2STModel = Literal["hgb", "logistic"]


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (base 2) between two discrete distributions."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        nz = a > 0
        return float(np.sum(a[nz] * np.log2(a[nz] / b[nz])))

    return max(0.0, 0.5 * kl(p, m) + 0.5 * kl(q, m))


def overlap(p: np.ndarray, q: np.ndarray) -> float:
    """Histogram overlap: sum of the element-wise minimum of the two distributions."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return float(np.minimum(p / p.sum(), q / q.sum()).sum())


def energy_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Energy distance ``2E|A-B| - E|A-A'| - E|B-B'|`` (Euclidean, 0 for equal distributions)."""
    ab = pairwise_distances(a, b).mean()
    aa = pairwise_distances(a).mean()
    bb = pairwise_distances(b).mean()
    return float(2 * ab - aa - bb)


def median_bandwidth(z: np.ndarray, rng: np.random.Generator, n: int = 2000) -> float:
    """Median pairwise Euclidean distance of a random subsample (RBF bandwidth heuristic)."""
    s = z[rng.choice(len(z), size=min(n, len(z)), replace=False)]
    d = pairwise_distances(s)
    return float(np.median(d[np.triu_indices(len(s), 1)]))


def mmd2(a: np.ndarray, b: np.ndarray, bandwidth: float) -> float:
    """Unbiased squared maximum mean discrepancy with an RBF kernel of the given bandwidth."""
    gamma = 1.0 / (2 * bandwidth**2)

    def k(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.exp(-gamma * pairwise_distances(u, v, metric="sqeuclidean"))

    kaa, kbb, kab = k(a, a), k(b, b), k(a, b)
    m, n = len(a), len(b)
    return float(
        (kaa.sum() - np.trace(kaa)) / (m * (m - 1))
        + (kbb.sum() - np.trace(kbb)) / (n * (n - 1))
        - 2 * kab.mean()
    )


def random_directions(dim: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` random unit vectors in ``dim`` dimensions, as a ``(dim, n)`` matrix."""
    v = rng.normal(size=(dim, n))
    return v / np.linalg.norm(v, axis=0, keepdims=True)


_QUANTILES = np.linspace(0.005, 0.995, 100)


def sliced_wasserstein(pa: np.ndarray, pb: np.ndarray) -> float:
    """Sliced 1-Wasserstein distance from data already projected on shared directions.

    Args:
        pa, pb: ``(n, n_directions)`` projections of the two samples.

    Returns:
        The mean over directions of the 1-D Wasserstein-1 distance, approximated on a
        grid of quantiles. In the units of the embedding space.
    """
    qa = np.quantile(pa, _QUANTILES, axis=0)
    qb = np.quantile(pb, _QUANTILES, axis=0)
    return float(np.abs(qa - qb).mean())


def _classifier(model: C2STModel, seed: int) -> Any:
    if model == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.1, class_weight="balanced", random_state=seed
        )
    return make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")
    )


def c2st(
    z: np.ndarray,
    target: np.ndarray,
    cv_groups: np.ndarray,
    model: C2STModel,
    n_folds: int,
    seed: int,
) -> np.ndarray:
    """Classifier two-sample test: out-of-fold probability that each cell is in group 1.

    Folds hold out whole spatial blocks (``cv_groups``), so a classifier cannot score
    well by memorising neighbouring, near-identical cells. With identical distributions
    the out-of-fold AUC is 0.5.

    Returns:
        Out-of-fold probabilities (NaN where a fold could not be trained).
    """
    proba = np.full(len(target), np.nan)
    n_splits = min(n_folds, len(np.unique(cv_groups)))
    if n_splits < 2:
        return proba
    for tr, te in GroupKFold(n_splits=n_splits).split(z, target, cv_groups):
        if len(np.unique(target[tr])) < 2:
            continue
        clf = _classifier(model, seed)
        clf.fit(z[tr], target[tr])
        proba[te] = clf.predict_proba(z[te])[:, 1]
    return proba


def auc_and_tv(target: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    """AUC and the total-variation lower bound ``|2 * balanced_accuracy - 1|``.

    Any classifier's balanced accuracy bounds the total variation distance between the
    two class-conditional distributions from below, so the second value reads as "at
    least this share of the two distributions does not overlap".
    """
    ok = ~np.isnan(proba)
    t, p = target[ok], proba[ok]
    if len(np.unique(t)) < 2:
        return float("nan"), float("nan")
    auc = float(roc_auc_score(t, p))
    tv = abs(2 * float(balanced_accuracy_score(t, p > 0.5)) - 1)
    return auc, tv


def knn_radius(ref: np.ndarray, k: int, ref_blocks: np.ndarray | None = None) -> np.ndarray:
    """Distance from each ``ref`` point to its k-th nearest neighbour in ``ref``.

    With ``ref_blocks``, neighbours in the point's own spatial block are ignored.
    Neighbouring cells are nearly identical, so without this the radius mostly
    measures spatial autocorrelation rather than how common a landscape is.
    """
    if ref_blocks is None:
        k = min(k, len(ref) - 1)
        return np.asarray(NearestNeighbors(n_neighbors=k + 1).fit(ref).kneighbors(ref)[0][:, k])
    radius = np.full(len(ref), np.inf)
    n_query = min(len(ref), k + 64)
    dist, nb = NearestNeighbors(n_neighbors=n_query).fit(ref).kneighbors(ref)
    other_block = ref_blocks[nb] != ref_blocks[:, None]
    enough = other_block.sum(axis=1) >= k
    for i in np.flatnonzero(enough):
        radius[i] = dist[i][other_block[i]][k - 1]
    # Points whose nearest candidates are all in their own block: search the other blocks.
    for blk in np.unique(ref_blocks[~enough]):
        pts = np.flatnonzero(~enough & (ref_blocks == blk))
        pool = ref[ref_blocks != blk]
        if len(pool) >= k:
            radius[pts] = NearestNeighbors(n_neighbors=k).fit(pool).kneighbors(ref[pts])[0][:, -1]
    return radius


def coverage(
    ref: np.ndarray, other: np.ndarray, k: int, ref_blocks: np.ndarray | None = None
) -> np.ndarray:
    """Which ``ref`` points have an ``other`` point inside their k-nearest-neighbour ball.

    The ball's radius is the distance from the point to its k-th nearest neighbour in
    ``ref`` itself (outside its own spatial block if ``ref_blocks`` is given), so
    "covered" means: the other sample has a cell at least as similar as the reference
    sample's own k-th most similar cell elsewhere (the *coverage* of Naeem et al.,
    2020). With equal sample sizes and identical distributions the expected share of
    covered points is about ``1 - 0.5**k``.

    Returns:
        Boolean mask over ``ref``.
    """
    radius = knn_radius(ref, k, ref_blocks)
    nearest = NearestNeighbors(n_neighbors=1).fit(other).kneighbors(ref)[0][:, 0]
    return np.asarray(nearest <= radius)


def halfplane_split(x: np.ndarray, y: np.ndarray, angle_deg: float, share: float) -> np.ndarray:
    """Split points by a straight line: the ``share`` of points furthest "behind" it.

    The line is perpendicular to the direction ``angle_deg`` (0 = east, 90 = north) and
    placed so that ``share`` of the points fall on the side opposite that direction.

    Returns:
        Boolean mask of the points on that side.
    """
    a = np.deg2rad(angle_deg)
    s = x * np.cos(a) + y * np.sin(a)
    return np.asarray(s <= np.quantile(s, share))
