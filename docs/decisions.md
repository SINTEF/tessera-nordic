# Design decisions

This file records what we found out about the upstream data and the choices
built on those findings. Update it when a decision changes.

## 1. geotessera API findings (inspected 2026-11-27, geotessera 0.11.1)

Findings come from reading the installed package source
(`geotessera/core.py`, `registry.py`, `store.py`, `zarr.py`) and from probing
real tiles over Trøndelag, Bergen and Jämtland. They were **not** taken from
older documentation, which describes a different API.

### Two access paths

geotessera 0.11 exposes two independent ways to get embeddings:

| | `GeoTessera` (NPY tiles) | `GeoTesseraZarr` (Zarr store) |
|---|---|---|
| Unit | 0.1° × 0.1° tile, downloaded as `.npy` files | arbitrary region read from a cloud Zarr store, one group per UTM zone |
| Default dataset | v1.1 variant **`cambridge`** ("Cambridge test run") | v1.1 variant **`dclimate`** ("Complete global run") |
| Upstream status | *deprecated* in favour of Zarr/Icechunk (docstring of `registry.Dataset`) | recommended |
| Local files | `<embeddings_dir>/global_0.1_degree_representation/<year>/grid_<lon>_<lat>/…npy` + `_scales.npy`, and `global_0.1_degree_tiff_all/grid_<lon>_<lat>.tiff` | none (optional on-disk chunk cache via `cache_dir`) |

The two variants are separate inference runs. Upstream states that
**embeddings from different variants do not interoperate**, even at the same
version. A run must therefore use one variant throughout.

**Coverage matters.** Counting NPY tiles per 1° cell for 2024 shows that the
`cambridge` NPY tiles cover Norway's mainland, but almost none of Sweden
(for example 0 tiles in the 1° cell holding Östersund, 0 tiles east of 12° E
at 63° N). The `dclimate` Zarr store returned valid embeddings at Östersund,
Jämtland (12.5° E), Stockholm, Kiruna and Trondheim.

**Decision:** the default backend is `zarr` (v1.1 `dclimate`). The `npy`
backend (v1.1 `cambridge`) is kept as an option, because it is the classic
"download a tile, process it, delete it" path and it works for Norway-only
areas. Both backends go through the same per-tile processing.

### Listing tiles for a bbox and year

- NPY: `GeoTessera(...).registry.iter_tiles_in_region(bounds, year)` (or the
  list version `load_blocks_for_region`) yields `(year, tile_lon, tile_lat)`
  for tile **centres**. The query is padded by 0.05°, so a bbox whose edges
  lie exactly on tile edges also returns the neighbouring ring of tiles.
  We therefore enumerate tiles ourselves (every 0.1° box with a non-zero-area
  intersection with the area) and, for the NPY backend, keep only tiles that
  the registry lists.
- Zarr: there is no public tile listing. We enumerate the same 0.1° grid and
  read each tile's box with `GeoTesseraZarr.read_region_quantized(bbox, year)`.
  A box over open sea comes back with all-non-finite scales and produces
  zero valid pixels.
- Tile centres are at `x.x5` degrees; `registry.tile_from_world(lon, lat)`
  snaps coordinates to them.
- Available years: 2017–2025 in the Zarr store; 2015–2025 in the NPY
  manifest (2015–2016 are sparse). 2024 is fully covered in our area.

### Quantisation

- Raw storage is `int8 (H, W, 128)` plus `float32 scales (H, W)` (one scale
  per pixel, broadcast over the 128 bands). Dequantised value =
  `int8 * scale` (`geotessera.dequantize_embedding`).
- `GeoTessera.fetch_embedding()` and `GeoTesseraZarr.read_region()` return
  **already dequantised** `float32 (H, W, 128)`.
- We read the quantised form (`np.load` of the NPY files, or
  `read_region_quantized`) and dequantise one band at a time while
  accumulating statistics. This keeps peak memory per worker at about
  100 MB instead of 300+ MB for a float32 tile.

### CRS and transform

- Every tile is delivered in its **native UTM zone** (EPSG:326xx), 10 m
  pixels, north-up. Zones follow the standard 6° rule: geotessera does
  **not** apply the Norway/Svalbard exceptions (a Bergen tile at 5.35° E is
  EPSG:32631, not 32632).
- NPY: CRS and transform are read from the landmask GeoTIFF of the tile
  (`_get_utm_projection_from_landmask`). The tile origin is not aligned to a
  10 m lattice (for example `x0 = 564716.23`).
- Zarr: each zone is one continuous grid aligned to 10 m (for example
  `x0 = 564720.0`); `read_region*` returns the window's transform and the
  zone CRS. A bbox that crosses a zone seam is served from the centre zone
  only; 0.1° tiles never cross a 6° seam, so this does not affect us.
- **A tile's pixel array is the UTM envelope of its lon/lat box**, so
  neighbouring tiles overlap at their corners (a 0.1° tile at 63.45° N is
  about 1125 × 522 pixels). To avoid counting a pixel twice we keep only
  pixels whose centre, converted to lon/lat, falls inside the tile's own
  half-open box `[lon0, lon0+0.1) × [lat0, lat0+0.1)`. This assigns every
  10 m pixel to exactly one tile.

### Land mask

- NPY: a `uint8` GeoTIFF per tile, `0 = water`, `1 = land`, nodata 0.
  `fetch_embedding()` does **not** apply it. We apply it when
  `apply_landmask: true`.
- Zarr: water and gaps are encoded in the scales: `NaN` = water, `+inf` =
  no data, finite = valid. Some stores mark NaN as no-data instead (store
  attribute `geotessera:mask_source`); either way a non-finite scale means
  "not a valid pixel".
- **Observed behaviour:** for the tiles we probed (Trondheimsfjord, the
  outer coast north of Bergen) the landmask was **all ones** and the
  embeddings were finite over fjord and near-shore sea. Only tiles entirely
  over open sea are missing (Zarr `probe` returns `nodata`). In practice
  fjords, lakes and coastal sea are **embedded** and will appear in the
  clustering, most likely as their own water clusters. See the open
  questions in the README.

### Embedding version

- `GeoTessera.dataset_version` (`"v1.1"`), `GeoTessera.dataset_variant`
  (`"cambridge"`), `GeoTessera.version` (the library version). The
  embeddings directory gets a `tessera_metadata.json` sidecar with the
  dataset path and variant.
- `GeoTesseraZarr.dataset` (a `Dataset(version, variant, ...)`),
  `.model_version` (`"https://geotessera.org/model/1.1"`) and
  `.build_version` (store build, `"0.10.3"` at inspection time).
- We record all of these, plus the geotessera version, in each run's
  `provenance.json`.

### Speed observed

One 0.1° tile (about 75 MB int8 + 2.3 MB scales) took 8–10 s to fetch over
both backends on a home/office connection; one NPY fetch took 99 s, so
throughput varies a lot. This is the basis of the size estimates in the
configs and README.

## 2. Target grid

- **CRS:** EPSG:3035 (ETRS89 / LAEA Europe). Equal-area, so every 1 km cell
  has the same area across Norway and Sweden; it is also the CRS of the EEA
  reference grid.
- **Cells:** aligned to multiples of the cell size (1000 m by default) in
  EPSG:3035 coordinates, which is the alignment the EEA grid uses. The cell
  size is set in the config.
- **Cell id:** an `int64` packing the column and row index
  (`(ix + 2^30) << 32 | (iy + 2^30)`, with `ix = floor(x / size)`).
  A readable `cell_code` in EEA style is stored too, for example
  `1kmE4321N3210` (the lower-left corner divided by the cell size).
- **Pixel to cell:** every valid 10 m pixel centre is projected from its
  UTM zone to EPSG:3035 with `pyproj` and assigned to the cell that contains
  it. No resampling of the embeddings happens.

## 3. Aggregation method

- Per tile and cell we store **mergeable sufficient statistics**: valid
  pixel count, the sum of each of the 128 dimensions and the sum of squares
  (float64). A cell that straddles tile boundaries gets contributions from
  several tiles; summing the statistics and only then dividing gives exactly
  the same mean and standard deviation as aggregating all its pixels at once.
  Averaging per tile and then averaging the averages would not.
- `merge` sums the statistics per cell and derives
  `mean = sum / n` and `std = sqrt(max(sumsq / n - mean^2, 0))` (population
  standard deviation).
- Cells with fewer than `min_pixel_count` valid pixels are dropped (a full
  1 km cell has 10 000 pixels at 10 m). With a boundary polygon, cells whose
  centre falls outside it are dropped too.
- **Known limitation:** mean pooling blurs mixed landscapes. A cell that is
  half lake and half spruce forest gets a mean vector that matches neither,
  and may fall into a cluster that represents neither. The per-cell `std`
  shows how mixed a cell is, but does not fix it.
- **Future option:** cluster at 10 m (for example MiniBatchKMeans on a
  sample of pixels, then predict every pixel), then aggregate *per-cell
  class fractions* instead of means. This keeps mixtures explicit and fits
  the same per-tile, mergeable pattern (fractions are counts / n).

## 4. Execution model

- Work is split into 0.1° tiles. Each tile writes one small Parquet file of
  cell statistics to a **stats cache** under
  `${TN_DATA_DIR}/stats/<source key>/`, plus a line in `manifest.jsonl`.
  The cache key holds the dataset version, variant, year, grid CRS, cell
  size and land-mask setting, so different settings never mix.
- The cache is deliberately **outside** the timestamped run directory, so
  an interrupted `aggregate` continues where it stopped, and later runs
  (for example a new k sweep) reuse finished tiles. The run directory
  records which cache it used.
- Tiles are processed by a process pool (`spawn` start method, safe on both
  macOS and Linux). Failed tiles are retried with exponential backoff and
  jitter; tiles that still fail are reported and the command exits non-zero
  so that a rerun picks them up.

## 5. Version pinning

- Python is pinned to 3.12 (`.python-version`, `requires-python`).
- `geotessera` is pinned to an exact version (`==0.11.1`) because its API
  changed substantially between releases. Upgrading it means re-checking
  section 1.
- All other dependencies have lower bounds in `pyproject.toml` and are
  locked exactly in the committed `uv.lock`. CI and the Docker image install
  with `uv sync --frozen`.
- The TESSERA dataset is pinned in each config (`dataset_version`,
  `dataset_variant`) and recorded with the model and store build versions in
  `provenance.json`.

## 6. Comparing the two countries' distributions

- **Question:** are Norway's and Sweden's embeddings distributed similarly? We answer
  it with one clustering fitted on both countries together, then compare each
  country's cluster histogram. Clustering each country separately and matching
  centroids was rejected because the match itself would add an uncertain step.
- **Measures:** Jensen–Shannon divergence (base 2) and histogram overlap between the
  cluster shares. Because those depend on `k` and on one clustering (seed ARI
  0.5–0.65), the stage also compares the countries directly in the embedding space
  (`pipeline/distribution.py`): a classifier two-sample test (gradient boosting,
  5-fold CV holding out 50 km blocks), energy distance, kernel MMD² (one RBF bandwidth
  for all comparisons), sliced Wasserstein-1, and k-NN coverage ("does the other
  country have a cell as similar as my own k-th most similar cell elsewhere?").
- **Coverage radius:** a cell's k-th nearest neighbour is almost always an adjacent
  cell, so the first version said 43% of Norway had "no counterpart" in the other
  half of Norway. Neighbours in the cell's own 50 km block are now excluded from the
  radius (baseline: 12–15% uncovered).
- **Reference scale:** "two random halves of one country" only measures sampling
  noise; it does not say whether a difference is large *for two regions*. The stage
  therefore also cuts the area by straight lines at 24 angles: the pooled area at the
  Norway/Sweden size ratio, and each country into halves. The real pair's percentile
  among those splits says whether the national border separates the landscape more
  than an arbitrary line does.
- **Uncertainty:** neighbouring 1 km cells are strongly correlated, so naive per-cell
  resampling would give far too narrow intervals. We resample whole 50 km blocks
  (`compare.block_size`).
- **Baseline:** a divergence is only meaningful relative to natural variation. We
  compare the JSD between a random half of each country with the JSD between two
  random halves of the *same* country, at the same sample size, because finite-sample
  JSD is biased upwards.
- **Sampling:** `aggregate.sample_fraction` draws a seeded random subset of tiles
  (reproducible and resumable). A 5% sample (~790 tiles) estimates each country's
  distribution at about 1/20 of the full run's download.
- **Border analysis (2026-11-27, exploratory):** on a strip across the border at
  63.0–63.8° N, a country classifier reached 92% balanced accuracy overall but
  separated the real border no better than placebo lines drawn inside one country,
  and neighbouring cells differed no more across the border than within a country.
  Locally, differences between the countries reflect landscape, not a border effect.

## 7. Water filter (ESA WorldCover)

- **Why:** the v1.1 land mask does not remove water (section 1), so fjords, lakes and
  near-shore sea inside the country polygons are embedded and clustered like land.
  Norway has far more fjord and coast than Sweden, so part of any country difference
  would simply be "more water".
- **Source:** ESA WorldCover 2021 v200 (10 m, CC BY 4.0), the most detailed open
  Europe-wide water layer that is published as Cloud-Optimised GeoTIFFs. Each 0.1°
  TESSERA tile reads only its own 1200 × 1200 pixel window (about 0.4 s); 3° files
  over open sea do not exist and give empty statistics.
- **Same tiles:** work units carry a `sample_key` (`<lon>_<lat>`) and
  `aggregate.sample_fraction` samples by that key, so both sources sample exactly the
  same tiles. The TESSERA sample is unchanged (checked against the 2026-11-27 run).
- **Per cell, not per pixel:** WorldCover becomes per-cell class fractions (a separate
  feature source, not used for clustering), and `merge.max_water_fraction` drops cells
  above the threshold. Masking per pixel would need WorldCover reprojected onto every
  TESSERA tile's UTM grid inside the TESSERA source and would invalidate the existing
  statistics cache; per-cell filtering keeps sources independent. Cells under the
  threshold still average in their water pixels (at most 20% with the default).
- **Year:** WorldCover 2021 vs TESSERA 2024. Permanent water changes little in three
  years; the class fractions are otherwise only used to describe clusters.

### Result on the 5% sample (2026-11-28, before the water filter)

43 300 cells (NO 17 560, SE 25 740), 14.6 min for the comparison stage on a laptop.

| Statistic | NO vs SE | Halves of one country (NO / SE) | Pooled line splits: median (range) | Percentile of NO vs SE |
|---|---|---|---|---|
| C2ST AUC | 0.960 | 0.490 / 0.487 | 0.980 (0.94–0.99) | 21 |
| Energy distance | 0.123 | 0.006 / 0.006 | 0.146 (0.04–0.24) | 42 |
| Sliced Wasserstein | 0.059 | 0.013 / 0.013 | 0.061 (0.03–0.08) | 46 |
| Uncovered NO / SE | 0.68 / 0.73 | 0.13–0.15 / 0.12 | 0.71 / 0.72 | 29 / 58 |
| JSD, k=8 | 0.190 | 0.01 | 0.242 (0.05–0.40) | 29 |

Reading:

- The two countries are clearly *not* identically distributed. A classifier tells
  them apart with AUC 0.96, against 0.49 for two halves of one country.
- But **any** straight line through Fennoscandia gives two regions that are just as
  separable (AUC 0.94–0.99). The embeddings carry a strong regional signature
  (climate, latitude, relief), so "separable" is the norm for contiguous regions.
- Against that reference, Norway vs Sweden sits at the 20th–45th percentile on most
  statistics: the border separates the landscape **less** than a typical line of the
  same size ratio. North-to-south lines (angles near 90°) separate most.
- Halves of Norway are more alike than Norway vs Sweden (NO vs SE above every
  within-Norway split on SWD and JSD). Halves of Sweden are usually *more* different
  than Norway vs Sweden, because Sweden spans a larger north–south gradient.
- Coverage: about 70% of cells lack a counterpart at k=5 in the other country, but
  the same holds for any two contiguous regions (0.55–0.84). At 1 km, cells carry
  regional detail that does not repeat elsewhere. Clusters 5 and 6 (mountains, far
  north) are well covered from Sweden's side but not from Norway's: Sweden's mountain
  cells look Norwegian, while most Norwegian mountain cells have no Swedish match.
