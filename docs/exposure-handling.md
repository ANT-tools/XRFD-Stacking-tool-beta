# Exposure-aware stacking

Version 0.7.1 treats exposure time as explicit per-frame metadata rather than assuming every TIFF has the same integration time.

## Automatic detection

The default automatic policy is **filename only**. Explicit unit-bearing filename tokens are recognized, for example:

```text
_5s_      -> 5.0 s
_10s_     -> 10.0 s
_0.5s_    -> 0.5 s
_500ms_   -> 0.5 s
```

Bare scan/frame numbers are never interpreted as exposure values.

Rayonix/MarCCD private-header `exposure_time` and `integration_time` values are read as diagnostics. The header exposure can be adopted explicitly or enabled as an automatic fallback, but it is not trusted by default because acquisition filenames and detector headers can disagree.

## Source priority

Manual and CSV assignments are persistent overrides. For non-overridden rows, the selectable policies are:

```text
filename only
filename -> header
header -> filename
```

The recommended policy for current Rayonix data is `filename only`, followed by manual assignment for rows without an explicit time token.

## Stacking normalization

When exposure normalization is enabled, each frame is rescaled to a common reference exposure before the selected stack estimator is evaluated:

```text
normalized_frame = frame * reference_exposure / frame_exposure
```

This assumes that the input frames are already on a comparable quantitative intensity basis. If dark/flat correction is required, use the correction workflow deliberately; do not double-correct data that were already corrected during detector acquisition.

## QC

The file table reports selected exposure, source, header exposure, and QC state. Filename/header disagreements are highlighted. Exposure-normalized stacking asks for confirmation when such a mismatch is present. Unknown exposures prevent exposure-normalized stacking until values are assigned.

## Provenance

Stack reports record the selected exposure, its source, filename exposure candidate, detector-header exposure candidate, MarCCD integration-time diagnostic, mismatch status, and normalization factor for each frame.
