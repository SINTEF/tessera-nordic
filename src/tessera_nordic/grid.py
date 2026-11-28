"""The analysis grid: square cells in a projected CRS, aligned to multiples of the cell size.

With the defaults (EPSG:3035, 1000 m) this matches the alignment of the EEA reference grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from affine import Affine
from pyproj import CRS, Transformer

# Indices are shifted to be non-negative; |index| < 2**30 keeps the packed id positive.
_OFFSET = np.int64(2**30)
_MASK = np.int64(2**32 - 1)


@lru_cache(maxsize=32)
def transformer(src_crs: str, dst_crs: str) -> Transformer:
    """Return a cached, always-xy transformer between two CRSs."""
    return Transformer.from_crs(
        CRS.from_user_input(src_crs), CRS.from_user_input(dst_crs), always_xy=True
    )


@dataclass(frozen=True)
class Grid:
    """A square grid in a projected CRS.

    Attributes:
        crs: CRS of the grid, e.g. ``"EPSG:3035"``.
        cell_size: Cell size in CRS units (metres).
    """

    crs: str
    cell_size: int

    def cell_index(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Column and row index of the cells containing points ``(x, y)`` (grid CRS)."""
        ix = np.floor(np.asarray(x, dtype=np.float64) / self.cell_size).astype(np.int64)
        iy = np.floor(np.asarray(y, dtype=np.float64) / self.cell_size).astype(np.int64)
        return ix, iy

    @staticmethod
    def encode(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
        """Pack column/row indices into one positive ``int64`` cell id.

        ``cell_id = (ix + 2**30) << 32 | (iy + 2**30)``; valid for ``|ix|, |iy| < 2**30``.
        """
        ix = np.asarray(ix, dtype=np.int64)
        iy = np.asarray(iy, dtype=np.int64)
        return ((ix + _OFFSET) << 32) | ((iy + _OFFSET) & _MASK)

    @staticmethod
    def decode(cell_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Inverse of :meth:`encode`."""
        cid = np.asarray(cell_id, dtype=np.int64)
        ix = (cid >> 32) - _OFFSET
        iy = (cid & _MASK) - _OFFSET
        return ix, iy

    def cell_ids(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Cell ids of the cells containing points ``(x, y)`` (grid CRS)."""
        return self.encode(*self.cell_index(x, y))

    def centres(self, cell_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Cell-centre coordinates (grid CRS) for the given ids."""
        ix, iy = self.decode(cell_id)
        half = self.cell_size / 2
        return ix * self.cell_size + half, iy * self.cell_size + half

    def cell_codes(self, cell_id: np.ndarray) -> np.ndarray:
        """EEA-style codes, e.g. ``1kmE4321N3210`` (lower-left corner / cell size)."""
        ix, iy = self.decode(cell_id)
        size = self.cell_size
        label = f"{size // 1000}km" if size % 1000 == 0 else f"{size}m"
        return np.array([f"{label}E{a}N{b}" for a, b in zip(ix, iy, strict=True)], dtype=object)

    def raster_for(self, cell_id: np.ndarray) -> tuple[Affine, int, int, np.ndarray, np.ndarray]:
        """Raster geometry covering the given cells.

        Returns:
            ``(transform, height, width, rows, cols)`` where ``rows``/``cols`` locate each
            cell in the raster.
        """
        ix, iy = self.decode(cell_id)
        ix0, ix1, iy0, iy1 = ix.min(), ix.max(), iy.min(), iy.max()
        width = int(ix1 - ix0 + 1)
        height = int(iy1 - iy0 + 1)
        transform = Affine(
            self.cell_size,
            0,
            float(ix0 * self.cell_size),
            0,
            -self.cell_size,
            float((iy1 + 1) * self.cell_size),
        )
        rows = (iy1 - iy).astype(np.int64)
        cols = (ix - ix0).astype(np.int64)
        return transform, height, width, rows, cols
