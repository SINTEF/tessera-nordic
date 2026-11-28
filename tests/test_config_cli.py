"""Config validation, the shipped configs, environment handling and the CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from tessera_nordic import paths
from tessera_nordic.cli import app
from tessera_nordic.config import RunConfig, load_config

CONFIGS = sorted((Path(__file__).parents[1] / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=[p.stem for p in CONFIGS])
def test_shipped_configs_validate(path: Path) -> None:
    cfg = load_config(path)
    assert cfg.run_name == path.stem
    assert cfg.year == 2024
    assert cfg.grid.crs == "EPSG:3035"


def test_expected_configs_exist() -> None:
    assert {p.stem for p in CONFIGS} >= {"smoke", "pilot", "full"}


def _base() -> dict:
    return {"run_name": "x", "area": {"name": "a", "bbox": [10, 63, 11, 64]}, "cluster": {"k": 5}}


def test_single_k_becomes_list() -> None:
    assert RunConfig.model_validate(_base()).cluster.k == [5]


def test_unknown_keys_are_rejected() -> None:
    data = _base() | {"workerz": 3}
    with pytest.raises(ValidationError):
        RunConfig.model_validate(data)


def test_area_needs_exactly_one_definition() -> None:
    data = _base()
    data["area"] = {"name": "a"}
    with pytest.raises(ValidationError):
        RunConfig.model_validate(data)


def test_boundary_path_expands_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _base()
    data["area"] = {"name": "a", "boundary": {"path": "${TN_DATA_DIR}/b.gpkg"}}
    cfg = RunConfig.model_validate(data)
    monkeypatch.delenv("TN_DATA_DIR", raising=False)
    with pytest.raises(ValueError, match="Unresolved"):
        cfg.area.boundary.resolved_path()  # type: ignore[union-attr]
    monkeypatch.setenv("TN_DATA_DIR", "/tmp/somewhere")
    assert str(cfg.area.boundary.resolved_path()) == "/tmp/somewhere/b.gpkg"  # type: ignore[union-attr]


def test_missing_environment_variable_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TN_OUTPUT_DIR", raising=False)
    with pytest.raises(paths.MissingEnvironmentError, match="TN_OUTPUT_DIR"):
        paths.output_dir()


def test_run_dirs_are_timestamped_and_latest_is_found(env_dirs) -> None:
    first = paths.new_run_dir("demo")
    second = paths.new_run_dir("demo")
    assert first != second and first.parent == second.parent
    assert paths.latest_run_dir("demo") == second


def test_cli_help_lists_every_stage() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for stage in ("aggregate", "merge", "cluster", "evaluate", "map", "compare", "run", "plan"):
        assert stage in result.output


def test_cli_plan_counts_smoke_tiles(env_dirs) -> None:
    smoke = next(p for p in CONFIGS if p.stem == "smoke")
    result = CliRunner().invoke(app, ["plan", str(smoke)])
    assert result.exit_code == 0, result.output
    assert ": 4 tiles" in result.output
