# Stacking algorithms in XRD Image Toolkit 0.5.0

The toolkit exposes several estimators because no single stack is optimal for every XRFD dataset.

## Recommended default: mean

Use **mean** when repeated accumulations are comparable, aligned, similarly exposed, and do not contain substantial transient artifacts. It preserves the ordinary arithmetic intensity scale and has the simplest statistical interpretation.

## Sum

Use **sum** when total accumulated detector counts are specifically desired. Be careful when comparing stacks containing different numbers of frames or different exposure times.

## Median

Use **median** as a robust diagnostic when isolated spikes or occasional outliers are present. It strongly suppresses extremes but is not an accumulated-count estimator and is less statistically efficient than the mean for clean Gaussian-like noise.

## Sigma-clipped mean

Iteratively rejects values that differ strongly from the per-pixel median/MAD estimate, then averages the survivors. Useful for cosmic-ray-like spikes, transient hot pixels, and occasional extreme detector events. Inspect the result against the ordinary mean to ensure real weak reflections are not being clipped.

## Trimmed mean

Drops a chosen fraction from both tails of the per-pixel distribution before averaging. It is less aggressive than a median while remaining resistant to repeated outliers. It works best when the stack contains enough frames for the trim fraction to be meaningful.

## Winsorized mean

Instead of deleting tail values, clamps them to lower/upper percentile limits and averages. It retains every frame but reduces the influence of extremes. This is useful as a moderate robust alternative when a trimmed mean feels too aggressive.

## Min/max rejected mean

For each pixel, rejects the single lowest and single highest finite value when at least four contributors are present. This is simple and effective for one-off bright or dark detector artifacts. It is especially interpretable for modest stack sizes.

## Inverse-variance weighted mean

Estimates a frame-level high-frequency noise sigma and weights each frame approximately by `1/sigma^2`. This can improve SNR when otherwise comparable frames have genuinely different noise levels.

Do **not** use noise weighting as a substitute for exposure-time, incident-flux, monitor-count, or transmission normalization. Those physical normalizations must be handled separately.

## Huber robust mean

Uses an iterative M-estimator: small residuals retain full weight while large residuals are smoothly down-weighted. It is a useful compromise between an arithmetic mean and hard clipping. It is computationally heavier than mean/min-max stacking but less discontinuous than sigma clipping.

## Suggested XRFD comparison workflow

For a new dataset, first build a **mean** stack. Then compare it with **sigma-clipped mean**, **min/max rejected mean**, and optionally **Huber mean**. If the robust products only remove isolated detector events while genuine diffraction features remain unchanged, they are reasonable alternatives. If weak spots or layer lines change substantially, inspect the contributing frames rather than automatically choosing the cleaner-looking stack.

Temporal stacks and symmetry/quadrant folding solve different problems. Temporal stacking combines repeated experimental exposures; quadrant folding combines symmetry-related measurements within a pattern. Keep those steps and their metadata separate.
