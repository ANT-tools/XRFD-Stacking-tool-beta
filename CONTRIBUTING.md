# Contributing

Thank you for improving the XRD Image Toolkit.

## Development setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Before submitting changes

Run:

```bash
python -m compileall xrd_toolkit
pytest
```

For performance-sensitive changes, also run:

```bash
python benchmarks/benchmark_xrd_toolkit.py --size 2048
```

## Scientific invariants

Please preserve these design rules:

1. Raw TIFF files must never be overwritten.
2. Quantitative detector-coordinate data must remain separate from display-only transforms.
3. Log/sqrt/gamma/asinh/tone-curve operations must not silently alter quantitative corrected TIFF exports.
4. Spatial operations such as registration, cropping, flipping, rotation, interpolation, or symmetry remapping must be recorded in provenance.
5. Detector-native outputs intended for an existing PONI calibration must preserve the original detector row/column coordinate convention.
6. Robust stacking methods should be documented as estimators rather than literal accumulated photon-count images where appropriate.
7. GPU code must retain a reliable CPU fallback.

## Data and privacy

Do not commit confidential experimental data, raw detector TIFFs, sample identifiers, or proprietary calibration files unless they are explicitly approved for public distribution.

Tests should use small synthetic arrays generated in code.

## Repository organization

- `xrd_toolkit/` — application/library code
- `tests/` — lightweight regression tests using synthetic data
- `benchmarks/` — performance scripts
- `docs/` — user/developer documentation
- `.github/workflows/` — CI

Keep top-level files limited to entry points, package metadata, licensing, contribution guidance, and the main README.

## Pull requests

Prefer focused pull requests. Describe:

- what changed
- whether quantitative arrays can change
- whether detector coordinates can change
- whether provenance fields changed
- CPU/GPU behavior
- tests performed
