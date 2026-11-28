# tessera-nordic: runs any CLI stage. Data lives in mounted volumes, never in the image.
#
#   docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t tessera-nordic .
#   docker run --rm -v /srv/tn/data:/data -v /srv/tn/outputs:/outputs \
#       -v /srv/tn/tessera:/tessera tessera-nordic run configs/smoke.yaml
FROM python:3.12-slim

# rasterio/pyogrio/pyproj wheels bundle GDAL, PROJ and GEOS; these are their remaining
# system dependencies (expat for GDAL's XML drivers, gomp for scikit-learn's OpenMP).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libexpat1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.10.2 /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    TN_DATA_DIR=/data \
    TN_OUTPUT_DIR=/outputs \
    TESSERA_DATA_DIR=/tessera

WORKDIR /app

# Dependencies first (cached layer), then the project itself.
COPY pyproject.toml uv.lock .python-version README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY configs ./configs
RUN uv sync --frozen --no-dev

ARG GIT_COMMIT=unknown
ENV TN_GIT_COMMIT=${GIT_COMMIT}

RUN useradd --create-home --uid 1000 tn \
    && mkdir -p /data /outputs /tessera \
    && chown tn:tn /data /outputs /tessera
USER tn
VOLUME ["/data", "/outputs", "/tessera"]

ENTRYPOINT ["tessera-nordic"]
CMD ["--help"]
