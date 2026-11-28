"""The feature-source abstraction.

A feature source turns some input dataset into per-cell sufficient statistics on the
shared analysis grid. Work is split into independent *units* (for TESSERA: 0.1-degree
tiles) so that the aggregate stage can run them in parallel, resume after interruption
and merge results by cell id. A future climate or terrain source implements the same
interface and is joined to the others on ``cell_id`` in the merge stage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from shapely.geometry.base import BaseGeometry

from tessera_nordic.grid import Grid
from tessera_nordic.stats import CellStats


@dataclass(frozen=True)
class WorkUnit:
    """One independent piece of work for a source.

    Attributes:
        unit_id: Stable, filename-safe identifier (used for the manifest and output file).
        params: JSON-serialisable parameters the source needs to process the unit.
    """

    unit_id: str
    params: dict[str, Any] = field(default_factory=dict)


class FeatureSource(ABC):
    """Produces per-cell statistics of ``n_features`` features for one input dataset."""

    #: Short name, used as the column prefix in the merged table (``<name>_mean_000``).
    name: str
    #: Number of features per pixel.
    n_features: int

    @abstractmethod
    def cache_key(self) -> str:
        """Identifier of everything that changes the statistics (dataset, year, grid...).

        Statistics are cached under this key, so it must change whenever results would.
        """

    @abstractmethod
    def list_units(self, area: BaseGeometry) -> list[WorkUnit]:
        """Units of work that together cover ``area`` (EPSG:4326)."""

    @abstractmethod
    def process_unit(self, unit: WorkUnit, grid: Grid) -> CellStats:
        """Fetch the data of one unit and reduce it to per-cell statistics on ``grid``."""

    def cleanup_unit(self, unit: WorkUnit) -> None:  # noqa: B027 - optional hook
        """Remove raw downloads of ``unit`` (called unless ``keep_raw`` is set)."""

    def discard_unit(self, unit: WorkUnit) -> None:  # noqa: B027 - optional hook
        """Remove possibly corrupt raw files of ``unit`` before a retry."""

    @abstractmethod
    def provenance(self) -> dict[str, Any]:
        """Versions and settings of the input data, recorded with every run."""
