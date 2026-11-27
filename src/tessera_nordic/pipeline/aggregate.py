"""Stage 1 - aggregate: stream tiles and reduce them to per-cell statistics.

Each unit (tile) writes ``<stats dir>/tiles/<unit_id>.parquet`` and appends a line to
``<stats dir>/manifest.jsonl``. The stats directory lives in ``${TN_DATA_DIR}`` and is
keyed by :meth:`FeatureSource.cache_key`, so reruns skip finished tiles and an
interrupted run continues where it stopped.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import random
import time
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np

from tessera_nordic import paths
from tessera_nordic.area import load_area_geometry
from tessera_nordic.config import AggregateConfig, RunConfig
from tessera_nordic.grid import Grid
from tessera_nordic.sources.base import FeatureSource, WorkUnit
from tessera_nordic.stats import write_parquet_atomic

log = logging.getLogger(__name__)

SUMMARY_FILE = "aggregate_summary.json"


class AggregateIncompleteError(RuntimeError):
    """Some units failed after all retries; rerun the command to retry them."""


def stats_dir(source: FeatureSource) -> Path:
    """``${TN_DATA_DIR}/stats/<cache key>``."""
    return paths.data_dir() / "stats" / source.cache_key()


def unit_path(sdir: Path, unit_id: str) -> Path:
    """Parquet file holding the statistics of one unit."""
    return sdir / "tiles" / f"{unit_id}.parquet"


class Manifest:
    """Append-only JSON-lines record of completed units."""

    def __init__(self, directory: Path) -> None:
        """Open (or create) ``directory/manifest.jsonl``."""
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.path = directory / "manifest.jsonl"

    def completed(self) -> dict[str, dict[str, Any]]:
        """Records of completed units whose output file still exists."""
        done: dict[str, dict[str, Any]] = {}
        if not self.path.exists():
            return done
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:  # torn last line after a crash
                    continue
                if (
                    rec.get("status") == "done"
                    and unit_path(self.directory, rec["unit_id"]).exists()
                ):
                    done[rec["unit_id"]] = rec
        return done

    def append(self, record: dict[str, Any]) -> None:
        """Append one record and flush it to disk."""
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


def process_with_retry(
    source: FeatureSource, unit: WorkUnit, grid: Grid, sdir: Path, cfg: AggregateConfig
) -> dict[str, Any]:
    """Process one unit, retrying with exponential backoff and jitter.

    Returns:
        A manifest record with ``status`` ``"done"`` or ``"failed"``.
    """
    attempt = 0
    start = time.monotonic()
    while True:
        try:
            stats = source.process_unit(unit, grid)
            write_parquet_atomic(stats.to_frame(), unit_path(sdir, unit.unit_id))
            if not cfg.keep_raw:
                source.cleanup_unit(unit)
            return {
                "unit_id": unit.unit_id,
                "status": "done",
                "n_cells": len(stats),
                "n_pixels": int(stats.count.sum()),
                "seconds": round(time.monotonic() - start, 2),
                "attempts": attempt + 1,
            }
        except Exception as exc:
            attempt += 1
            try:
                source.discard_unit(unit)
            except Exception:  # cleanup must not hide the real error
                log.debug("discard_unit failed for %s", unit.unit_id, exc_info=True)
            if attempt > cfg.max_retries:
                log.error("Unit %s failed after %d attempts: %r", unit.unit_id, attempt, exc)
                return {"unit_id": unit.unit_id, "status": "failed", "error": repr(exc)}
            delay = cfg.retry_backoff_s * 2 ** (attempt - 1) * (0.5 + random.random())
            log.warning(
                "Unit %s attempt %d/%d failed (%r); retrying in %.0fs",
                unit.unit_id,
                attempt,
                cfg.max_retries + 1,
                exc,
                delay,
            )
            time.sleep(delay)


def sample_units(units: Sequence[WorkUnit], fraction: float, seed: int) -> list[WorkUnit]:
    """A reproducible random sample of ``units``.

    The draw depends only on the unit ids and the seed, so a rerun selects the same
    units (and resumes them) regardless of listing order.
    """
    ordered = sorted(units, key=lambda u: u.unit_id)
    n = max(1, round(fraction * len(ordered)))
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(ordered), size=n, replace=False))
    return [ordered[i] for i in idx]


# --- worker-process state (spawn start method) --------------------------------------

_WORKER: dict[str, Any] = {}


def _init_worker(config_json: dict[str, Any], bounds: tuple[float, ...], queue: Any) -> None:
    from tessera_nordic.logging_utils import configure_worker_logging
    from tessera_nordic.sources import build_sources

    configure_worker_logging(queue)
    config = RunConfig.model_validate(config_json)
    _WORKER["config"] = config
    _WORKER["sources"] = build_sources(config, bounds)  # type: ignore[arg-type]
    _WORKER["grid"] = Grid(config.grid.crs, config.grid.cell_size)


def _worker_process(source_index: int, unit: WorkUnit, sdir: str) -> dict[str, Any]:
    config: RunConfig = _WORKER["config"]
    source = _WORKER["sources"][source_index]
    return process_with_retry(source, unit, _WORKER["grid"], Path(sdir), config.aggregate)


# --- driver ---------------------------------------------------------------------------


class _Progress:
    def __init__(self, total: int, label: str) -> None:
        self.total = total
        self.label = label
        self.done = 0
        self.failed = 0
        self.start = time.monotonic()
        self.last = 0.0

    def update(self, ok: bool) -> None:
        self.done += ok
        self.failed += not ok
        now = time.monotonic()
        finished = self.done + self.failed
        if now - self.last > 30 or finished == self.total:
            self.last = now
            elapsed = now - self.start
            rate = finished / elapsed if elapsed > 0 else 0.0
            eta = (self.total - finished) / rate if rate > 0 else float("nan")
            log.info(
                "%s: %d/%d tiles (%d failed), %.2f tiles/min, ETA %.1f min",
                self.label,
                finished,
                self.total,
                self.failed,
                rate * 60,
                eta / 60,
            )


def _run_units(
    source: FeatureSource,
    source_index: int,
    units: Sequence[WorkUnit],
    config: RunConfig,
    sdir: Path,
    manifest: Manifest,
    bounds: tuple[float, float, float, float],
    workers: int,
) -> list[dict[str, Any]]:
    grid = Grid(config.grid.crs, config.grid.cell_size)
    progress = _Progress(len(units), source.name)
    records: list[dict[str, Any]] = []

    def handle(rec: dict[str, Any]) -> None:
        records.append(rec)
        if rec["status"] == "done":
            manifest.append(rec)
        progress.update(rec["status"] == "done")

    if workers <= 1:
        for unit in units:
            handle(process_with_retry(source, unit, grid, sdir, config.aggregate))
        return records

    from tessera_nordic.logging_utils import start_queue_listener

    queue, listener = start_queue_listener()
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker,
            initargs=(config.model_dump(mode="json"), bounds, queue),
        ) as pool:
            # Submit lazily so memory stays flat for tens of thousands of tiles.
            pending: dict[Future[dict[str, Any]], WorkUnit] = {}
            it = iter(units)
            for unit in it:
                pending[pool.submit(_worker_process, source_index, unit, str(sdir))] = unit
                if len(pending) >= workers * 2:
                    break
            while pending:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for fut in finished:
                    unit = pending.pop(fut)
                    try:
                        handle(fut.result())
                    except Exception as exc:  # worker crashed (e.g. out of memory)
                        log.error("Worker crashed on %s: %r", unit.unit_id, exc)
                        handle({"unit_id": unit.unit_id, "status": "failed", "error": repr(exc)})
                    nxt = next(it, None)
                    if nxt is not None:
                        pending[pool.submit(_worker_process, source_index, nxt, str(sdir))] = nxt
    finally:
        listener.stop()
    return records


def run_aggregate(
    config: RunConfig,
    run_dir: Path,
    sources: list[FeatureSource] | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    """Aggregate every unit of every enabled source; skip units already done.

    Args:
        config: Run configuration.
        run_dir: Run directory; receives ``aggregate_summary.json``.
        sources: Feature sources (built from the config if omitted).
        workers: Override ``config.aggregate.workers``.

    Returns:
        The summary that was written.

    Raises:
        AggregateIncompleteError: if some units failed after all retries.
    """
    geom = load_area_geometry(config.area)
    bounds: tuple[float, float, float, float] = tuple(geom.bounds)
    if sources is None:
        from tessera_nordic.sources import build_sources

        sources = build_sources(config, bounds)
    n_workers = workers or config.aggregate.workers
    summary: dict[str, Any] = {"area": config.area.name, "sources": {}}
    any_failed = False
    for index, source in enumerate(sources):
        sdir = stats_dir(source)
        manifest = Manifest(sdir)
        units = source.list_units(geom)
        if config.aggregate.sample_fraction < 1:
            units = sample_units(units, config.aggregate.sample_fraction, config.seed)
            log.info(
                "%s: random sample of %d units (fraction %.3f, seed %d)",
                source.name,
                len(units),
                config.aggregate.sample_fraction,
                config.seed,
            )
        if config.aggregate.max_tiles is not None:
            units = units[: config.aggregate.max_tiles]
        done = manifest.completed()
        todo = [u for u in units if u.unit_id not in done]
        log.info(
            "%s: %d units in area, %d already done, %d to process with %d worker(s); stats in %s",
            source.name,
            len(units),
            len(units) - len(todo),
            len(todo),
            n_workers,
            sdir,
        )
        started = time.monotonic()
        records = _run_units(source, index, todo, config, sdir, manifest, bounds, n_workers)
        failed = [r for r in records if r["status"] != "done"]
        any_failed |= bool(failed)
        done = manifest.completed()
        in_area = [u.unit_id for u in units]
        summary["sources"][source.name] = {
            "cache_key": source.cache_key(),
            "stats_dir": str(sdir),
            "n_units": len(units),
            "n_processed_now": len(records) - len(failed),
            "n_skipped_done": len(units) - len(todo),
            "n_failed": len(failed),
            "failed": failed,
            "n_pixels": int(sum(done[u].get("n_pixels", 0) for u in in_area if u in done)),
            "seconds": round(time.monotonic() - started, 1),
            "units": in_area,
        }
    (run_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if any_failed:
        raise AggregateIncompleteError(
            "Some tiles failed; see aggregate_summary.json. Rerun to retry only those."
        )
    return summary
