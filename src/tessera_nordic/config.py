"""Run configuration: one YAML file per run, validated with pydantic.

Nothing in the library hard-codes paths, areas, years or hyperparameters; they all
come from a :class:`RunConfig`. Strings in ``area.boundary.path`` may reference
environment variables (``${TN_DATA_DIR}/boundaries/...``).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_UNRESOLVED_VAR = re.compile(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?")


class _Strict(BaseModel):
    """Base model that rejects unknown keys, so typos in configs fail loudly."""

    model_config = ConfigDict(extra="forbid")


class BoundaryConfig(_Strict):
    """A vector boundary file (any format GeoPandas reads) and an attribute filter."""

    path: str = Field(description="Path to the vector file; ${ENV_VARS} are expanded.")
    column: str | None = Field(default=None, description="Attribute used to select features.")
    values: list[str] | None = Field(default=None, description="Values of `column` to keep.")
    clip_bbox: tuple[float, float, float, float] | None = Field(
        default=None,
        description="Optional (min_lon, min_lat, max_lon, max_lat) the boundary is clipped to, "
        "e.g. to drop overseas territories.",
    )

    @model_validator(mode="after")
    def _column_and_values(self) -> BoundaryConfig:
        if (self.column is None) != (self.values is None):
            raise ValueError("boundary.column and boundary.values must be given together")
        return self

    def resolved_path(self) -> Path:
        """Return the path with environment variables expanded.

        Raises:
            ValueError: if a referenced environment variable is not set.
        """
        expanded = os.path.expandvars(self.path)
        if _UNRESOLVED_VAR.search(expanded):
            raise ValueError(
                f"Unresolved environment variable in boundary path {self.path!r}; "
                "set it in your environment or .env file."
            )
        return Path(expanded).expanduser()


class AreaConfig(_Strict):
    """Study area: either a WGS84 bbox or a boundary polygon file."""

    name: str = Field(description="Human-readable area name, recorded in provenance.")
    bbox: tuple[float, float, float, float] | None = Field(
        default=None, description="(min_lon, min_lat, max_lon, max_lat) in EPSG:4326."
    )
    boundary: BoundaryConfig | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> AreaConfig:
        if (self.bbox is None) == (self.boundary is None):
            raise ValueError("area needs exactly one of `bbox` or `boundary`")
        if self.bbox is not None:
            min_lon, min_lat, max_lon, max_lat = self.bbox
            if not (-180 <= min_lon < max_lon <= 180 and -90 <= min_lat < max_lat <= 90):
                raise ValueError(f"invalid bbox {self.bbox}")
        return self


class TesseraConfig(_Strict):
    """Settings for the TESSERA embeddings feature source."""

    backend: Literal["zarr", "npy"] = Field(
        default="zarr",
        description="`zarr` streams from the v1.1 dclimate store (covers NO and SE); "
        "`npy` downloads deprecated per-tile files (v1.1 cambridge, Norway only).",
    )
    dataset_version: str = Field(default="v1.1", description="TESSERA embedding version.")
    dataset_variant: str = Field(
        default="dclimate", description="Inference run; variants do not interoperate."
    )
    store_url: str | None = Field(
        default=None, description="Override the Zarr store URL (zarr backend only)."
    )
    apply_landmask: bool = Field(
        default=True, description="Drop pixels flagged as water / no-data upstream."
    )


class GridConfig(_Strict):
    """Target analysis grid."""

    crs: str = Field(default="EPSG:3035", description="Projected CRS of the grid.")
    cell_size: int = Field(default=1000, gt=0, description="Cell size in CRS units (metres).")


class AggregateConfig(_Strict):
    """Streaming download + per-tile aggregation settings."""

    workers: int = Field(default=2, ge=1)
    keep_raw: bool = Field(
        default=False,
        description="Keep raw downloads (NPY tiles) or the Zarr chunk cache in TESSERA_DATA_DIR.",
    )
    max_retries: int = Field(default=4, ge=0, description="Retries per tile after a failure.")
    retry_backoff_s: float = Field(default=10.0, ge=0, description="Base backoff in seconds.")
    max_tiles: int | None = Field(
        default=None, ge=1, description="Process at most this many tiles (for testing)."
    )


class MergeConfig(_Strict):
    """Per-cell merge settings."""

    min_pixel_count: int = Field(default=2500, ge=1)
    clip_to_area: bool = Field(
        default=True, description="Drop cells whose centre lies outside the area geometry."
    )


class PCAConfig(_Strict):
    """PCA settings."""

    n_components: int = Field(default=16, ge=1)
    whiten: bool = False


class ClusterConfig(_Strict):
    """Clustering settings."""

    feature_prefixes: list[str] = Field(
        default_factory=lambda: ["tessera_mean_"],
        description="Columns of the merged table used as features, selected by prefix.",
    )
    l2_normalize: bool = True
    pca: PCAConfig = Field(default_factory=PCAConfig)
    method: str = Field(default="minibatch_kmeans", description="Name in the method registry.")
    k: list[int] = Field(description="A single k or a sweep, e.g. [5, 8, 12].")
    params: dict[str, Any] = Field(
        default_factory=lambda: {"batch_size": 4096, "n_init": 3, "max_iter": 300},
        description="Extra keyword arguments for the clustering method.",
    )

    @field_validator("k", mode="before")
    @classmethod
    def _k_as_list(cls, value: Any) -> Any:
        return [value] if isinstance(value, int) else value

    @field_validator("k")
    @classmethod
    def _k_valid(cls, value: list[int]) -> list[int]:
        if not value or any(k < 2 for k in value):
            raise ValueError("every k must be >= 2")
        return sorted(set(value))


class EvaluateConfig(_Strict):
    """Evaluation settings."""

    silhouette_sample: int = Field(default=20000, ge=100)
    stability_seeds: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])

    @field_validator("stability_seeds")
    @classmethod
    def _two_seeds(cls, value: list[int]) -> list[int]:
        if len(value) < 2:
            raise ValueError("stability needs at least two seeds")
        return value


class MapConfig(_Strict):
    """Map output settings."""

    dpi: int = Field(default=150, ge=50)


class SourcesConfig(_Strict):
    """Enabled feature sources. Add new sources here as optional fields."""

    tessera: TesseraConfig = Field(default_factory=TesseraConfig)


class RunConfig(_Strict):
    """A complete, validated run configuration."""

    run_name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    seed: int = 42
    year: int = Field(default=2024, ge=2015)
    area: AreaConfig
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    grid: GridConfig = Field(default_factory=GridConfig)
    aggregate: AggregateConfig = Field(default_factory=AggregateConfig)
    merge: MergeConfig = Field(default_factory=MergeConfig)
    cluster: ClusterConfig
    evaluate: EvaluateConfig = Field(default_factory=EvaluateConfig)
    map: MapConfig = Field(default_factory=MapConfig)


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a run configuration from a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        The validated configuration.
    """
    with Path(path).open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return RunConfig.model_validate(data)


def dump_config(config: RunConfig, path: str | Path) -> None:
    """Write the resolved configuration (defaults filled in) as YAML.

    Args:
        config: The configuration to write.
        path: Destination file.
    """
    data = config.model_dump(mode="json")
    with Path(path).open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
