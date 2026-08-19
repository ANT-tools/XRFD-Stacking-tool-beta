# XRD Image Toolkit

Desktop Python toolkit for processing CCD/area-detector X-ray fiber diffraction (XRFD) TIFF data.

Current version: **0.6.1**

## What it does

- load individual TIFFs or folders of repeated accumulations
- stack repeated frames with mean, sum, median, sigma-clipped, trimmed, Winsorized, min/max-rejected, inverse-variance, and Huber estimators
- optional translational frame registration
- dark, flat-field, and blank/background corrections
- detector masks for hot pixels, saturation, and beamstop regions
- CPU optimization plus optional NVIDIA/CuPy acceleration
- display-only contrast controls including log, sqrt, gamma, asinh, histogram equalization, and custom tone curves
- beam-center and fiber-axis estimation
- equatorial and meridional profile extraction
- quadrant folding / symmetry averaging with asymmetry diagnostics
- TIFF, PNG, CSV, JSON, and annotated provenance-text export

The toolkit deliberately separates **quantitative detector-coordinate data** from **display-only enhancement**. Raw TIFFs are never overwritten.

## Installation

Python 3.10+ is recommended.

### Standard CPU installation

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

### Optional GPU acceleration

See [docs/gpu-acceleration.md](docs/gpu-acceleration.md). GPU support is optional; the CPU path remains fully supported.

## Run

After an editable/package installation:

```bash
xrfd-toolkit
```

or:

```bash
python -m xrd_toolkit
```

The legacy launcher also remains available:

```bash
python run_xrd_toolkit.py
```

## Recommended XRFD workflow

```text
raw sample accumulations
        ↓
inspect / reject bad frames
        ↓
temporal stack
        ↓
dark / flat / blank corrections
        ↓
mask invalid detector regions
        ↓
beam-center + fiber-axis determination
        ↓
quantitative detector-coordinate product
        ├──────────────→ calibrated / structural analysis
        │
        └──→ display-only contrast / tone curve → PNG

optional after geometry validation:
quantitative corrected image
        ↓
quadrant folding / symmetry averaging
        ↓
meridian / equator / layer-line analysis
```

## Detector-coordinate safety

A corrected detector-coordinate TIFF is intended to retain the native detector row/column geometry unless a spatial operation such as registration, cropping, rotation, flipping, interpolation, or symmetry remapping is explicitly applied. The provenance sidecars record these operations.

Do **not** substitute a display PNG or symmetrized image for a native detector image when applying an existing PONI calibration.

## Provenance

Corrected TIFF exports can generate matching sidecars:

```text
sample_corrected.tif
sample_corrected.provenance.json
sample_corrected.provenance.txt
```

The JSON is the canonical machine-readable record. The annotated TXT contains the same typed values with explanatory comments. See [docs/provenance-format.md](docs/provenance-format.md).

## Documentation

- [GPU acceleration](docs/gpu-acceleration.md)
- [Performance notes](docs/performance.md)
- [Stacking algorithms](docs/stacking-algorithms.md)
- [Provenance format](docs/provenance-format.md)
- [Release notes: 0.6.1](docs/releases/0.6.1.md)
- [Detailed historical notes](docs/version-history.md)

## Repository layout

```text
xrd_toolkit/              Python package
benchmarks/               local performance benchmark
 tests/                   lightweight scientific-core tests
 docs/                    user/developer documentation
 .github/workflows/       continuous-integration checks
run_xrd_toolkit.py        legacy launcher
pyproject.toml            package metadata
requirements.txt          minimal runtime dependencies
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md). Scientific changes should preserve the distinction between quantitative intensity processing and display-only transformations.

## Data policy

Do not commit raw experimental TIFFs, detector dumps, generated stack arrays, or confidential sample data to this repository. Use synthetic or explicitly shareable test data only.

## License

See [LICENSE](LICENSE).
