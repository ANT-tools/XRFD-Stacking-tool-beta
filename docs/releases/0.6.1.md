# XRD Image Toolkit 0.6.1 — Provenance Sidecar Update

## Added

- Automatic provenance sidecars for corrected TIFF exports.
- Canonical JSON provenance.
- Annotated, machine-readable TXT provenance.
- Reference parser for the TXT format.
- SHA-256 checksum of an exported corrected TIFF.
- Explicit native-detector geometry flags.
- Stack-registration provenance.
- Quantitative correction provenance.
- TIFF-header preservation statement.
- Editable intended calibration label.
- Editable expected detector dimensions.
- PONI suitability summary flag.

## Default detector/calibration context

The Export tab defaults to:

- intended calibration label: `MD-033`
- expected detector rows: `3072`
- expected detector columns: `3072`

These values can be changed for other experiments.

## Important interpretation

`safe_detector_geometry_for_existing_poni = true` means:

1. the output array shape matches the expected detector dimensions; and
2. the corrected product has not been spatially rotated, flipped, cropped,
   interpolated, registered, or symmetry-remapped.

The toolkit still records the named calibration as **intended/unverified**
unless the calibration file itself is audited.

## TIFF headers

The toolkit still writes a new float32 TIFF using `tifffile`. Original TIFF
acquisition tags are not copied verbatim. The provenance record now states this
explicitly so later users do not assume that acquisition metadata were carried
into the exported TIFF header.
