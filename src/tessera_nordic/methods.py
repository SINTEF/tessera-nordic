"""Clustering methods behind a small interface, so GMM or others can be added later.

To add a method: subclass :class:`ClusterMethod` and register it in :data:`METHODS`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans


class ClusterMethod(ABC):
    """A clustering algorithm with a fixed number of clusters ``k``."""

    name: str

    def __init__(self, **params: Any) -> None:
        """Store method-specific parameters from the config (``cluster.params``)."""
        self.params = params

    @abstractmethod
    def fit(self, X: np.ndarray, k: int, seed: int) -> Any:
        """Fit a model with ``k`` clusters and return it (must be joblib-serialisable)."""

    @abstractmethod
    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Cluster labels ``0..k-1`` for ``X``."""

    def inertia(self, model: Any, X: np.ndarray) -> float | None:
        """Within-cluster sum of squares, if the method defines it."""
        return None


class MiniBatchKMeansMethod(ClusterMethod):
    """scikit-learn ``MiniBatchKMeans``."""

    name = "minibatch_kmeans"

    def fit(self, X: np.ndarray, k: int, seed: int) -> MiniBatchKMeans:
        """Fit MiniBatchKMeans."""
        model = MiniBatchKMeans(n_clusters=k, random_state=seed, **self.params)
        return model.fit(X)

    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Nearest-centroid labels."""
        return np.asarray(model.predict(X), dtype=np.int32)

    def inertia(self, model: Any, X: np.ndarray) -> float | None:
        """Inertia of ``X`` w.r.t. the fitted centroids (not the last mini-batch)."""
        return float(-model.score(X))


METHODS: dict[str, type[ClusterMethod]] = {MiniBatchKMeansMethod.name: MiniBatchKMeansMethod}


def get_method(name: str, params: dict[str, Any] | None = None) -> ClusterMethod:
    """Instantiate a registered method.

    Raises:
        KeyError: if ``name`` is not registered.
    """
    if name not in METHODS:
        raise KeyError(f"unknown clustering method {name!r}; available: {sorted(METHODS)}")
    return METHODS[name](**(params or {}))
