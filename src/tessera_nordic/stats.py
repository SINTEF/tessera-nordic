"""Mergeable per-cell sufficient statistics: count, sum and sum of squares.

Summing these over any partition of a cell's pixels (e.g. across tile borders) and
dividing only at the end gives exact means and standard deviations.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CELL_ID = "cell_id"
COUNT = "count"


def sum_col(i: int) -> str:
    """Column name of the sum of feature ``i``."""
    return f"sum_{i:03d}"


def sumsq_col(i: int) -> str:
    """Column name of the sum of squares of feature ``i``."""
    return f"sumsq_{i:03d}"


@dataclass
class CellStats:
    """Sufficient statistics for a set of cells.

    Attributes:
        cell_id: ``(n,)`` int64 cell ids, unique.
        count: ``(n,)`` int64 number of valid pixels.
        sums: ``(n, F)`` float64 per-feature sums.
        sumsq: ``(n, F)`` float64 per-feature sums of squares.
    """

    cell_id: np.ndarray
    count: np.ndarray
    sums: np.ndarray
    sumsq: np.ndarray

    @property
    def n_features(self) -> int:
        """Number of features."""
        return int(self.sums.shape[1])

    def __len__(self) -> int:
        return int(self.cell_id.shape[0])

    @classmethod
    def empty(cls, n_features: int) -> CellStats:
        """Statistics with no cells."""
        return cls(
            np.empty(0, np.int64),
            np.empty(0, np.int64),
            np.empty((0, n_features), np.float64),
            np.empty((0, n_features), np.float64),
        )

    def to_frame(self) -> pd.DataFrame:
        """Wide DataFrame: ``cell_id``, ``count``, ``sum_000..``, ``sumsq_000..``."""
        f = self.n_features
        data: dict[str, np.ndarray] = {CELL_ID: self.cell_id, COUNT: self.count}
        data.update({sum_col(i): self.sums[:, i] for i in range(f)})
        data.update({sumsq_col(i): self.sumsq[:, i] for i in range(f)})
        return pd.DataFrame(data)

    @classmethod
    def from_frame(cls, df: pd.DataFrame, n_features: int) -> CellStats:
        """Inverse of :meth:`to_frame`."""
        return cls(
            df[CELL_ID].to_numpy(np.int64),
            df[COUNT].to_numpy(np.int64),
            df[[sum_col(i) for i in range(n_features)]].to_numpy(np.float64),
            df[[sumsq_col(i) for i in range(n_features)]].to_numpy(np.float64),
        )


def accumulate(
    cell_id: np.ndarray,
    n_features: int,
    feature: Callable[[int], np.ndarray],
) -> CellStats:
    """Accumulate statistics for pixels already assigned to cells.

    Features are pulled one at a time through ``feature(i)`` so that callers can
    dequantise lazily and keep memory low.

    Args:
        cell_id: ``(N,)`` cell id of each valid pixel.
        n_features: Number of features ``F``.
        feature: Returns the ``(N,)`` values of feature ``i`` for the same pixels.

    Returns:
        Statistics for each distinct cell in ``cell_id``.
    """
    if cell_id.size == 0:
        return CellStats.empty(n_features)
    uniq, inverse = np.unique(cell_id, return_inverse=True)
    n = uniq.size
    count = np.bincount(inverse, minlength=n).astype(np.int64)
    sums = np.empty((n, n_features), np.float64)
    sumsq = np.empty((n, n_features), np.float64)
    for i in range(n_features):
        v = np.asarray(feature(i), dtype=np.float64)
        sums[:, i] = np.bincount(inverse, weights=v, minlength=n)
        sumsq[:, i] = np.bincount(inverse, weights=v * v, minlength=n)
    return CellStats(uniq.astype(np.int64), count, sums, sumsq)


def merge_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Sum statistics of the same cell across frames (e.g. across tiles)."""
    frames = [f for f in frames if len(f)]
    if not frames:
        raise ValueError("no statistics to merge")
    df = pd.concat(frames, ignore_index=True)
    return df.groupby(CELL_ID, sort=True).sum().reset_index()


def finalize(stats: CellStats) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell mean and population standard deviation.

    Returns:
        ``(mean, std)`` arrays of shape ``(n, F)``, float32.
    """
    n = np.maximum(stats.count, 1).astype(np.float64)[:, None]
    mean = stats.sums / n
    var = np.maximum(stats.sumsq / n - mean * mean, 0.0)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write ``df`` to Parquet via a temporary file and an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp, compression="zstd")
    tmp.replace(path)
