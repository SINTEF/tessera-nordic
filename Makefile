# Common tasks. Long runs (pilot, full) are best started inside tmux on a server.
SHELL := /bin/bash
UV ?= uv
RUN := $(UV) run --frozen tessera-nordic
IMAGE ?= tessera-nordic:latest
GIT_COMMIT := $(shell git rev-parse HEAD 2>/dev/null || echo unknown)

.PHONY: help install lint format test smoke pilot full plan-full docker-build

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-13s %s\n", $$1, $$2}'

install:  ## Create .venv from uv.lock and install pre-commit hooks
	$(UV) sync --frozen
	$(UV) run --frozen pre-commit install

lint:  ## ruff (lint + format check) and mypy
	$(UV) run --frozen ruff check src tests
	$(UV) run --frozen ruff format --check src tests
	$(UV) run --frozen mypy

format:  ## Auto-format and auto-fix
	$(UV) run --frozen ruff format src tests
	$(UV) run --frozen ruff check --fix src tests

test:  ## Unit tests (synthetic tiles, no network)
	$(UV) run --frozen pytest

smoke:  ## Whole pipeline on 2x2 real tiles (network required)
	$(RUN) run configs/smoke.yaml

pilot:  ## Pilot transect Trøndelag-Jämtland (network required)
	$(RUN) run configs/pilot.yaml

full:  ## Norway + Sweden (network required; needs the boundary file, see README)
	$(RUN) run configs/full.yaml

plan-full:  ## Count tiles / estimate volume for the full run (no download)
	$(RUN) plan configs/full.yaml

docker-build:  ## Build the container image
	docker build --build-arg GIT_COMMIT=$(GIT_COMMIT) -t $(IMAGE) .
