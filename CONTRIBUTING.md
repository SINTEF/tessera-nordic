# Contributing

## Setup

```bash
git clone git@github.com:SINTEF/tessera-nordic.git
cd tessera-nordic
make install          # uv sync --frozen + pre-commit install
cp .env.example .env  # then point the three directories OUTSIDE the repository
```

You need [uv](https://docs.astral.sh/uv/) ≥ 0.10. uv installs Python 3.12 if it is
missing.

## Before you push

```bash
make lint   # ruff check, ruff format --check, mypy
make test   # pytest; synthetic tiles only, no network
```

The pre-commit hooks run ruff and mypy on every commit (`make install` enables them).
They use the tool versions locked in `uv.lock`, so the hooks, `make lint` and CI agree.

## Rules for this repository

- **Code only.** The repository is public. Never commit data, rasters, Parquet/Zarr
  outputs, boundary files, `.env` or credentials. `.gitignore` blocks the usual
  extensions and the `check-added-large-files` hook rejects files over 500 kB, but
  check `git status` anyway.
- **No hard-coded paths, areas, years or hyperparameters.** They belong in a config
  in `configs/` (validated by `tessera_nordic.config`).
- **No `print` in library code.** Use `logging.getLogger(__name__)`.
- **Tests must not touch the network.** Use the synthetic-tile fixtures in
  `tests/conftest.py`.
- Public functions get type hints and a docstring.
- If you upgrade `geotessera`, re-check the findings in `docs/decisions.md` first.

## Branches and pull requests

1. Branch from `main`: `feature/<topic>`, `fix/<topic>`, `docs/<topic>` or
   `setup/<topic>`.
2. Keep commits small and logical, with an imperative subject line, for example
   `feat: add terrain feature source`.
3. Open a pull request to `main`. CI (lint, type check, tests, Docker build) must
   pass before merging.
4. Describe in the PR what changed, why, and how you tested it. For pipeline
   changes, say whether you ran `make smoke` and include its runtime.

## Adding a feature source

Implement `tessera_nordic.sources.base.FeatureSource` (`cache_key`, `list_units`,
`process_unit`, `provenance`), add its settings to `SourcesConfig` in `config.py` and
register it in `tessera_nordic.sources.build_sources`. The aggregate and merge stages
handle the rest: statistics are cached per unit and joined to the other sources on
`cell_id`.
