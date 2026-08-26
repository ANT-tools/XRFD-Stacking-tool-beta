# XRD Image Toolkit

Desktop Python toolkit for processing and analyzing CCD/area-detector X-ray fiber diffraction (XRFD) TIFF data.

Current version: **0.7.2**

## 0.7.2 interface hotfix — scrollable control tabs

All control tabs now use vertically scrollable content areas. A visible scrollbar is available at the right edge of every tab, and the mouse wheel/trackpad scrolls the tab while the pointer is anywhere over its controls. This prevents lower options from becoming inaccessible on standard desktop displays or when display scaling makes the GUI taller than the available screen.

The change is interface-only; it does not alter diffraction arrays, stacking algorithms, correction order, layer-line analysis, provenance, or exported detector coordinates.


## What it does

- load individual TIFFs or folders of repeated accumulations
- stack repeated frames with mean, sum, median, sigma-clipped, trimmed, Winsorized, min/max-rejected, inverse-variance, exposure-weighted, and Huber estimators
- normalize mixed-integration-time datasets before stacking
- optional translational frame registration
- optional dark, flat-field, and blank/background corrections
- detector masks for hot pixels, saturation, and beamstop regions
- CPU optimization plus optional NVIDIA/CuPy acceleration
- display-only contrast controls including log, sqrt, gamma, asinh, histogram equalization, and custom tone curves
- beam-center and fiber-axis estimation
- equatorial and meridional profile extraction
- layer-line ladder fitting and selected-layer multi-peak fitting
- non-destructive circle, ellipse, rectangle, and line ROI annotations
- image comparison using side-by-side, difference, ratio, and overlay views
- quadrant folding / symmetry averaging with asymmetry diagnostics
- analysis-session save/load
- TIFF, PNG, CSV, JSON, and annotated provenance-text export

The toolkit deliberately separates **quantitative detector-coordinate data** from **display-only enhancement**. Raw TIFFs are never overwritten.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
```

You can also install the core dependencies directly:

```bash
python -m pip install -r requirements.txt
```

For optional GPU acceleration, see [docs/gpu-acceleration.md](docs/gpu-acceleration.md).

## Run

```bash
xrfd-toolkit
```

or:

```bash
python -m xrd_toolkit
```

The legacy launcher remains available:

```bash
python run_xrd_toolkit.py
```

## Recommended XRFD workflow

```text
sample TIFF accumulations
        ↓
inspect / reject bad frames
        ↓
optional exposure normalization
        ↓
temporal stack
        ↓
optional dark / flat / blank corrections
        ↓
mask invalid detector regions
        ↓
beam-center + fiber-axis determination
        ↓
quantitative detector-coordinate product
        ├──────────────→ calibration / integration / model comparison
        │
        ├──────────────→ layer-line / reflection analysis
        │
        └──→ display-only contrast / tone curve → PNG

optional after geometry validation:
quantitative corrected image
        ↓
quadrant folding / symmetry averaging
```

## New in 0.7.1: exposure parsing and file-selection QC

Version 0.7.1 hardens mixed-exposure stacking for Rayonix/MarCCD datasets.

- explicit filename exposure tokens such as `_5s_`, `_10s_`, `_0.5s_`, and `_500ms_` are parsed automatically
- bare run/frame numbers are never interpreted as exposure times
- the Rayonix/MarCCD private header `exposure_time` and `integration_time` values are read as diagnostics
- the default automatic policy is **filename only**, because detector-header timing can disagree with acquisition filenames
- alternative policies `filename → header` and `header → filename` are available
- filename/header disagreements are highlighted and produce a confirmation before exposure-normalized stacking
- manual exposure assignment supports seconds or milliseconds and can be applied to selected rows or all rows
- manual/CSV values can be cleared, selected rows can be re-detected, and header values can be explicitly confirmed
- the file table shows selected exposure, source, header exposure, and exposure-QC state
- `Ctrl+Shift+click` now explicitly adds a contiguous file range to the existing highlighted selection on all supported platforms
- exposure source/candidate information is retained in stack CSV/JSON/provenance records and analysis sessions

The recommended policy for the current Rayonix workflow is **filename only**, with manual assignment for files that do not contain an explicit exposure token.

## New in 0.7.0

### Exposure-aware stacking

The file table now tracks integration time. The toolkit attempts to read exposure time from TIFF metadata and supports manual or CSV assignment when metadata are absent. When exposure normalization is enabled, each frame is converted to counts equivalent to a user-selected reference integration time before the chosen stack estimator is applied.

A new **exposure-weighted mean** is available for comparable repeated measurements where longer exposures should contribute more statistical weight.

The input correction state can also be documented as already dark/flat corrected, raw/uncorrected, or custom/mixed. This is provenance metadata and does not silently apply corrections.

### Layer-line analysis

The Layer Lines tab supports an equator-to-anchor ladder. You can click an anchor reflection or let the toolkit suggest one, specify how many lines lie between the equator and anchor, locally refine the line positions against the experimental intensity profile, inspect residuals, export the fitted table, and extract a selected layer-line profile.

Selected layer-line profiles can be fit with multiple Gaussian components plus a linear background. The output includes peak centers, widths, integrated intensities, uncertainties, and R². This is useful for testing whether a broad arc is actually a combination of nearby reflections.

### ROI / shapes

Circles, ellipses, rectangles, and lines can be drawn directly on the diffraction image. These objects are non-destructive annotations. A selected ROI can report pixel count, mean, median, sum, standard deviation, and extrema, and ROI definitions can be saved/loaded as JSON.

### Image comparison

Two patterns can be compared as current corrected images, TIFFs, or stacks. Available modes include difference, absolute difference, ratio, overlay, and side-by-side display. B can be scaled to A by least-squares fit, total intensity, exposure time, a manual factor, or not at all. Correlation and RMS diagnostics are reported, and a selected ROI can be measured at the same detector coordinates in both images.

### Session files

Session JSON saves major analysis state—file inclusion/exposure assignments, stacking configuration, fiber geometry, contrast/tone-curve settings, layer-line settings, and ROI shapes—without embedding raw detector pixels.

## Detector-coordinate safety

A corrected detector-coordinate TIFF is intended to retain the native detector row/column geometry unless a spatial operation such as registration, cropping, rotation, flipping, interpolation, or symmetry remapping is explicitly applied. The provenance sidecars record these operations.

Layer-line profile extraction uses interpolation internally for analysis only; it does **not** rotate or resample the corrected TIFF.

Do **not** substitute a display PNG or symmetrized image for a native detector image when applying an existing PONI calibration.

## Provenance

Corrected TIFF exports can generate matching sidecars:

```text
sample_corrected.tif
sample_corrected.provenance.json
sample_corrected.provenance.txt
```

The JSON is the canonical machine-readable record. The annotated TXT contains the same typed values with explanatory comments. Exposure normalization and per-frame integration times are included in stack/provenance records.

See [docs/provenance-format.md](docs/provenance-format.md).

## Documentation

- [Exposure-aware stacking](docs/exposure-handling.md)
- [GPU acceleration](docs/gpu-acceleration.md)
- [Performance notes](docs/performance.md)
- [Stacking algorithms](docs/stacking-algorithms.md)
- [Provenance format](docs/provenance-format.md)
- [Release notes: 0.7.1](docs/releases/0.7.1.md)
- [Release notes: 0.7.0](docs/releases/0.7.0.md)
- [Detailed historical notes](docs/version-history.md)

## Repository layout

```text
xrd_toolkit/              Python package
benchmarks/               local performance benchmark
tests/                    synthetic scientific-core regression tests
docs/                     user/developer documentation
.github/workflows/        continuous-integration checks
run_xrd_toolkit.py        legacy launcher
pyproject.toml            package metadata
requirements.txt          minimal runtime dependencies
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md). Scientific changes should preserve detector-coordinate integrity and the distinction between quantitative intensity processing and display-only transformations.

## Data policy

Do not commit raw experimental TIFFs, detector dumps, generated stack arrays, or confidential sample data to this repository. Use synthetic or explicitly shareable test data only.

## License

See [LICENSE](LICENSE).
