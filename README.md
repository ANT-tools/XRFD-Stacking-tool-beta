# XRD Image Toolkit 0.6.1

Desktop GUI for CCD / area-detector X-ray diffraction TIFF processing, with
a new multi-accumulation stacking workflow for XRFD data.



## New in 0.6.0: custom tone curves and GPU robust-stack fix

The Contrast tab now includes an interactive custom tone-curve editor. The
curve maps normalized display input (X) to screen brightness (Y). Add, move
and remove control points directly on the graph, optionally constrain the
curve to remain monotonic, and save/load curves as JSON. The curve is
**display-only** and is never applied to quantitative diffraction intensities.

The GPU implementation of trimmed and Winsorized stacking no longer relies on
`cupy.nanpercentile`. A custom NaN-aware per-pixel GPU quantile is used
instead, and robust stacking automatically falls back to the CPU if a GPU
compatibility problem is encountered.

## New in 0.2.1: PNG sharing export

The Export tab now supports lightweight PNG output for sharing diffraction
images by email, in slides, or in manuscripts.

New controls:

- **Save current viewer as PNG** — exports whichever view is active:
  Raw, Corrected, Enhanced, Display, Mask, Background model, or Stack.
- **Save stack PNG** — exports a combined accumulation without first promoting
  it to the current working image.
- optional axes and tick labels
- optional plot title
- adjustable PNG DPI

For raw, corrected, enhanced and stacked images, the toolkit applies the
currently selected display transform and contrast settings when making the PNG.
That lets weak XRFD features remain visible despite the detector's large
dynamic range.

PNG is lossless as an image format, so it avoids JPEG block/ringing artefacts,
but the rendered PNG is still a **display/share product** and does not retain
the full floating-point detector dynamic range. Keep the TIFF for quantitative
analysis.


## New in 0.3.0: XRFD symmetry averaging / quadrant folding

The new **Symmetry** tab implements quadrant folding of a quantitatively
corrected fiber-diffraction image.

Recommended workflow:

```text
temporal TIFF accumulations
    → mean/sum stack
    → dark and blank/background corrections
    → masks
    → determine beam center + fiber axis
    → symmetry averaging / quadrant folding
    → meridian / equator / layer-line analysis
```

### Four-quadrant averaging

The toolkit transforms the corrected detector image into a local coordinate
system in which the equator is horizontal and the meridian/fiber axis is
vertical. At each output coordinate it samples the corresponding
symmetry-related detector locations and combines the valid measurements.

Available modes:

- four-quadrant
- centrosymmetric pair
- mirror across the meridian
- mirror across the equator

Mean and median combination are supported.

### Centered crop

A centered common crop is chosen automatically from the beam center, detector
dimensions and current fiber-axis angle. Custom equatorial and meridional
half-extents can also be entered.

### Mask-aware folding

The current quantitative mask is respected. A masked detector location is
excluded from the average while valid equivalent quadrants can still
contribute.

### Diagnostics

Before relying on the folded result, inspect:

- pairwise correlations between symmetry-related measurements
- the per-pixel asymmetry standard-deviation map
- the number of valid contributors at each output pixel
- normalized RMS asymmetry

Strong disagreement can indicate an inaccurate beam center, incorrect fiber
orientation, tilt/disorientation, detector or background artifacts, or
genuine sample asymmetry.

### SNR interpretation

For four equivalent quadrants with comparable independent random detector /
counting noise, the ideal SNR improvement from their mean is approximately:

```text
sqrt(4) = 2
```

If N repeated experimental accumulations are first averaged and then a valid
four-quadrant mean is performed, the ideal random-noise gain relative to one
quadrant of one accumulation is approximately:

```text
sqrt(4N) = 2*sqrt(N)
```

The four symmetry members are not four additional experimental samples. They
are symmetry-related measurements of the same reciprocal-space information.

### Symmetry exports

The Export tab can save:

- symmetrized floating-point TIFF
- symmetrized PNG
- asymmetry-standard-deviation TIFF
- JSON symmetry report with settings and pairwise correlations

The symmetrized pattern can also be selected as the source for the existing
meridional/equatorial profile analysis.


## New in 0.4.0: performance architecture

Version 0.4.0 targets large CCD images and repeated interactive processing.

Major changes:

- internal detector images are now normally `float32`, cutting working-memory
  bandwidth roughly in half relative to the previous float64 pipeline
- a staged `ProcessingEngine` cache separates quantitative correction, heavy
  enhancement, and display scaling, so changing log/sqrt/gamma display no
  longer reruns Gaussian/median filters or detector corrections
- fast sampled percentiles are used for display contrast by default
- the Matplotlib viewer decimates only the on-screen preview while preserving
  full-resolution arrays for analysis and export
- four-quadrant symmetry now resamples the detector into aligned coordinates
  once and constructs symmetry members by array flips, instead of performing
  a separate interpolation for each quadrant
- pairwise symmetry correlations are evaluated on a representative sampled
  grid instead of repeatedly scanning every detector pixel
- registration caches the reference FFT, uses real-valued FFTs, supports
  SciPy FFT worker threads, and can limit the FFT to a central registration
  crop
- TIFF decoding can use multiple workers
- broad Gaussian and median mathematical-background operations have optional fast
  reduced-grid approximations; exact full-resolution modes remain available
- optional NVIDIA CUDA acceleration is available through CuPy for the heavy
  filter, symmetry, FFT-registration, median-stack and sigma-clipping paths

The **Performance** tab controls the compute backend, viewer resolution,
fast percentile mode, median-background approximation, FFT crop/workers and
TIFF decode workers.

For GPU setup see `GPU_ACCELERATION.md`. To measure performance on your own
machine run `python benchmark_xrd_toolkit.py --size 2048`.

### Important numerical note

The quantitative image pipeline uses single-precision floating point for
speed and memory efficiency. Raw TIFFs are never overwritten. Stack mean/sum
accumulation still uses a float64 accumulator before the final float32 output.

The **fast broad Gaussian/median background** modes are approximations intended
for broad visual/background estimation. Disable the corresponding fast mode in
the Performance tab when an exact full-resolution filter is required.


## New in 0.5.0: geometry detection, contrast workbench, and expanded stacking

Version 0.5.0 implements the three priorities selected for the next XRFD update.

### 1. Stronger beam-center and fiber-axis detection

The Fiber Analysis tab now contains an automatic geometry detector. It works on
an internal, dynamic-range-compressed copy of the **quantitatively corrected**
image; it never alters the diffraction data.

The algorithm uses:

- 180-degree centrosymmetry to estimate/refine the beam center
- mirror-symmetry scoring to determine the two perpendicular pattern axes
- coarse-to-fine angular scanning
- an intensity-compressed detection image so the direct beam does not dominate
- the existing mask so detector defects and beamstop regions can be excluded
- a diagnostic score plot before the candidate is applied

The meridian/equator pair is mathematically ambiguous by 90 degrees when only
mirror symmetry is used. The toolkit therefore reports the equivalent axis
closest to the user's current fiber-axis estimate. The detected candidate is
**not applied automatically**; it can be reviewed and then accepted.

### 2. Dedicated Contrast tab

Contrast is now separated from spatial filtering. New display-only options are:

- linear, log, square-root, gamma, **asinh**, and global histogram-equalized tone curves
- legacy post-transform percentile scaling for backwards-compatible appearance
- source-intensity percentile levels
- manual black / white intensity levels
- robust MAD-based auto levels
- full-range levels
- optional black/white inversion
- optional adaptive local contrast normalization
- intensity histogram with current black/white markers
- a weak-reflection preset

All contrast operations are display-only. They do not alter the corrected TIFF,
symmetry input, line profiles, or structural-comparison intensity arrays.

### 3. Expanded stacking algorithms

The Stacking tab now includes:

- mean
- sum
- median
- sigma-clipped mean
- **trimmed mean**
- **Winsorized mean**
- **min/max rejected mean**
- **inverse-variance weighted mean**
- **Huber robust mean**

The inverse-variance method estimates each frame's high-frequency noise from
neighboring-pixel differences and uses frame-level `1/sigma^2` weights. Those
weights and the estimated noise are written to the stack report.

Recommended interpretation:

- **Mean** remains the default for clean, comparable repeated accumulations.
- **Sigma-clipped / min-max / trimmed / Huber** methods are useful diagnostics or
  robust alternatives when transient detector spikes or outlying pixels occur.
- **Inverse-variance weighting** is useful when comparable frames have different
  noise levels, but it should not be used as a substitute for exposure/monitor
  normalization when the incident intensity or exposure time changes.
- **Median / Winsorized / Huber** products should be treated as robust estimators,
  not literal accumulated photon-count images.

Robust per-pixel algorithms remain chunked and disk-backed so large CCD stacks
can be processed without loading the complete frame cube into RAM.

## 0.5.1 stacking reliability hotfix

The TIFF side panel now displays image dimensions and the stack builder checks
all selected dimensions before allocating buffers. Mixed detector/image sizes
can be excluded interactively. Windows memory-mapped temporary stack files are
also explicitly closed on success and failure.


## New in 0.6.1: provenance sidecars

Corrected detector-coordinate TIFF exports can now automatically create two
matching provenance files:

```text
sample_corrected.tif
sample_corrected.provenance.json
sample_corrected.provenance.txt
```

The JSON file is the canonical machine-readable record.

The TXT file contains the same typed data in an annotated section/key format.
Comments beginning with `#` explain what each value means for later diffraction
analysis. Data values are still encoded as JSON literals, which makes the text
format deterministic and machine readable.

Example:

```text
[coordinate_system]
# Meaning of NumPy/TIFF array axis 0.
axis_0 = "detector_row"

# True if a spatial rotation was applied before export.
rotation_applied = false
```

A reference parser is included:

```python
from xrd_toolkit.provenance import read_provenance_text

record = read_provenance_text("sample_corrected.provenance.txt")
print(record["coordinate_system"]["axis_0"])
```

The provenance record explicitly tracks:

- detector rows / columns
- array axis meaning
- upper-left array origin
- rotation / flip / crop / interpolation
- stack registration
- whether detector-native orientation remains intact
- dark / flat / background corrections
- monitor normalization
- detector masks
- stack method and input files
- beam center and fiber-axis angle
- intended geometry/PONI calibration label
- expected detector dimensions
- suitability for direct PONI geometry use
- TIFF header preservation status
- analysis guidance for corrected TIFF vs PNG vs symmetry products

The default calibration label is `MD-033` and the expected detector size is
3072 × 3072. These are editable in the Export tab.

The label records intended use; it does not independently verify a PONI file.

## Core design

The toolkit keeps distinct products separate:

1. **Individual raw TIFFs**
2. **Combined stack**
3. **Quantitatively corrected working image**
4. **Feature-enhanced image**
5. **Display-only transformed image**

The original TIFF files are never overwritten.

---

## New in 0.2.0: permanent TIFF side panel

When a folder is loaded, every `.tif` / `.tiff` file is displayed in a
permanent left-side pane.

Each row has an independent **Use** state.

You can:

- single-click a file to preview it
- Ctrl/Shift-select multiple rows
- include the selected rows
- exclude the selected rows
- include all
- exclude all
- invert the current inclusion set
- make only the currently highlighted rows part of the stack
- double-click a row to toggle inclusion
- press Space to toggle inclusion for highlighted rows

Excluded files stay visible and can still be previewed.

After a stack is built, the pane also displays:

- frame mean intensity
- frame maximum intensity
- correlation to the first included reference frame

---

## Stacking workflow

The **Stacking** tab combines only frames marked as included.

Available methods:

### Mean

Pixel-wise arithmetic mean of valid pixels.

Good default when repeated accumulations contain the same diffraction
pattern and random noise is the main limitation.

### Sum

Pixel-wise sum of valid counts.

Useful when preserving accumulated counts is desirable.

### Median

Pixel-wise median.

More resistant to isolated transient spikes, but does not preserve total
count statistics in the same way as mean/sum.

### Sigma-clipped mean

A robust mean that rejects large per-pixel deviations using a median/MAD
estimate before averaging.

Useful for transient spikes or occasional outlier pixels.

---

## Optional frame registration

The toolkit can align each included TIFF to the first included frame using
FFT phase correlation.

The current implementation estimates an **integer-pixel translational
shift** only.

Why integer shifts?

- no interpolation is required
- pixel values that remain inside the detector area are not resampled
- exposed edges are filled with NaN and ignored by stack statistics

A user-set maximum shift prevents an unexpectedly large registration from
being silently applied.

Registration should be used only if detector/beam drift is actually present.
If all accumulations are already aligned, leave it disabled.

---

## Large CCD stacks

Mean and sum are streamed frame-by-frame and do not require the whole stack
to be stored in memory.

Median and sigma-clipped mean require access to multiple frames at the same
pixel location. To avoid holding every full-resolution CCD frame in RAM, the
toolkit writes the selected frames to a temporary disk-backed NumPy array
and combines the image in row chunks.

The chunk height can be changed in the GUI.

---

## Stack products

After stacking, you can:

- preview the combined stack
- inspect QC values in the left file pane
- save the stacked image as a floating-point TIFF
- save a JSON stack report
- save a per-frame CSV report
- promote the stack to the **current working image**

Once promoted, the stack goes through the same processing pipeline as a
single TIFF:

raw-equivalent stack
→ detector/reference corrections
→ masking
→ enhanced copy
→ display copy
→ fiber-analysis tools

---

## Important scientific point

The stacking module currently combines the original TIFF accumulations.

Dark / flat / blank corrections are deliberately applied later when the
stack is promoted to the current working image. This prevents accidentally
applying the same correction twice.

If exposure time or incident monitor counts vary from frame to frame, those
per-frame normalization values should eventually be incorporated before
stacking. Version 0.2.0 assumes the included accumulations are comparable
exposures unless the user has independently verified otherwise.

---

## Existing processing features

### Quantitative / detector corrections

- dark subtraction
- flat-field correction
- blank/background image subtraction
- monitor/exposure normalization
- hot-pixel masking
- saturation masking
- circular beamstop mask

### Display / feature enhancement

- median filtering
- Gaussian smoothing
- broad Gaussian-background subtraction
- broad median-background subtraction
- Gaussian high-pass filtering
- unsharp masking
- linear display
- log display
- square-root display
- gamma display
- percentile contrast

### Fiber-analysis tools

- set beam center manually or by mouse click
- define fiber axis numerically or from two image clicks
- equator/fiber-axis overlays
- strip-averaged meridional profile
- strip-averaged equatorial profile
- profile peak detection
- profile CSV export

These profiles are currently in detector-pixel coordinates.

---

## Installation

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

Tkinter is included with most standard Windows Python installations.

---

## Run

From the toolkit directory:

```bash
python run_xrd_toolkit.py
```

---

## Recommended first stacking workflow

1. Open the folder containing repeated accumulations.
2. Preview several frames from the left pane.
3. Exclude frames with obvious sample movement, radiation damage, detector
   failure, saturation or an anomalous pattern.
4. Start with **mean** stacking.
5. Leave alignment off if the pattern is already stationary.
6. Build the stack.
7. Inspect the frame correlations in the side pane.
8. Compare mean, median and sigma-clipped mean if transient artefacts are
   present.
9. Save the stack and stack report.
10. Choose **Use stack as current working image**.
11. Continue with masking, background correction and fiber analysis.

For N comparable accumulations with independent random noise, a mean/sum
stack has an idealized SNR improvement of approximately `sqrt(N)` relative
to one frame.

---

## Planned next stage

- per-frame exposure / monitor metadata
- automatic frame-outlier detection
- stack preview statistics before committing
- subpixel registration option
- detector `.poni` calibration
- wavelength / distance / pixel-size metadata
- pixel → q / 2theta / d conversion
- q_parallel / q_perpendicular mapping
- sector integration
- calibrated meridian / equator profiles
- automated layer-line detection
- comparison against simulated PDB diffraction patterns
