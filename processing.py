from __future__ import annotations

from dataclasses import dataclass, asdict, fields
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, zoom

from .backend import resolve_backend


@dataclass
class ProcessingSettings:
    # Quantitative corrections
    dark_enabled: bool = False
    flat_enabled: bool = False
    background_enabled: bool = False
    background_scale: float = 1.0
    normalize_enabled: bool = False
    monitor_value: float = 1.0

    # Masks
    hot_pixels_enabled: bool = False
    hot_pixel_sigma: float = 8.0
    hot_pixel_size: int = 3
    saturation_enabled: bool = False
    saturation_value: float = 65535.0
    beamstop_enabled: bool = False
    beamstop_center_x: float = 0.0
    beamstop_center_y: float = 0.0
    beamstop_radius: float = 0.0

    # Enhancement / feature display
    median_filter_enabled: bool = False
    median_filter_size: int = 3
    gaussian_filter_enabled: bool = False
    gaussian_filter_sigma: float = 1.0
    gaussian_background_enabled: bool = False
    gaussian_background_sigma: float = 40.0
    median_background_enabled: bool = False
    median_background_size: int = 41
    high_pass_enabled: bool = False
    high_pass_sigma: float = 6.0
    unsharp_enabled: bool = False
    unsharp_sigma: float = 2.0
    unsharp_amount: float = 1.0

    # Display / contrast (display-only; quantitative intensities are unchanged)
    display_mode: str = "log"
    log_gain: float = 100.0
    gamma: float = 0.5
    asinh_strength: float = 20.0
    custom_curve_spec: str = "0,0;1,1"
    contrast_mode: str = "legacy percentile"
    percentile_low: float = 0.5
    percentile_high: float = 99.7
    manual_black: float = 0.0
    manual_white: float = 1.0
    robust_sigma: float = 6.0
    invert_display: bool = False
    local_contrast_enabled: bool = False
    local_contrast_sigma: float = 25.0
    local_contrast_strength: float = 0.45
    local_contrast_noise_floor: float = 0.15
    histogram_bins: int = 2048

    # Performance
    compute_backend: str = "auto"          # auto / cpu / gpu
    fast_percentiles: bool = True
    percentile_sample_target: int = 262144
    fast_gaussian_background: bool = True
    gaussian_background_downsample: int = 4
    fast_median_background: bool = True
    median_background_downsample: int = 4

    def to_dict(self):
        return asdict(self)


QUANTITATIVE_FIELDS = (
    "dark_enabled", "flat_enabled", "background_enabled", "background_scale",
    "normalize_enabled", "monitor_value", "hot_pixels_enabled", "hot_pixel_sigma",
    "hot_pixel_size", "saturation_enabled", "saturation_value", "beamstop_enabled",
    "beamstop_center_x", "beamstop_center_y", "beamstop_radius", "compute_backend",
)
ENHANCEMENT_FIELDS = (
    "median_filter_enabled", "median_filter_size", "gaussian_filter_enabled",
    "gaussian_filter_sigma", "gaussian_background_enabled", "gaussian_background_sigma",
    "median_background_enabled", "median_background_size", "high_pass_enabled",
    "high_pass_sigma", "unsharp_enabled", "unsharp_sigma", "unsharp_amount",
    "compute_backend", "fast_gaussian_background", "gaussian_background_downsample",
    "fast_median_background", "median_background_downsample",
)
DISPLAY_FIELDS = (
    "display_mode", "log_gain", "gamma", "asinh_strength", "custom_curve_spec", "contrast_mode",
    "percentile_low", "percentile_high", "manual_black", "manual_white",
    "robust_sigma", "invert_display", "local_contrast_enabled",
    "local_contrast_sigma", "local_contrast_strength", "local_contrast_noise_floor",
    "histogram_bins", "fast_percentiles", "percentile_sample_target", "compute_backend",
)


def _signature(settings, names):
    return tuple(getattr(settings, name) for name in names)


def validate_same_shape(reference, other, name):
    if other is not None and reference.shape != other.shape:
        raise ValueError(f"{name} has shape {other.shape}; expected {reference.shape}.")


def _as_float32(image):
    return np.asarray(image, dtype=np.float32)


def apply_dark(image, dark):
    validate_same_shape(image, dark, "Dark image")
    return image - np.asarray(dark, dtype=np.float32)


def apply_flat(image, flat, eps=1e-12):
    validate_same_shape(image, flat, "Flat image")
    flat = np.asarray(flat, dtype=np.float32)
    valid = np.isfinite(flat) & (flat > eps)
    if not np.any(valid):
        raise ValueError("Flat image contains no valid positive pixels.")
    norm = np.float32(np.nanmedian(flat[valid]))
    bad = (~np.isfinite(flat)) | (flat <= eps)
    out = np.asarray(image, dtype=np.float32).copy()
    # Algebraically image / (flat/norm), avoiding a full flat_norm temporary.
    out[~bad] *= norm / flat[~bad]
    out[bad] = np.nan
    return out, bad


def apply_background(image, background, scale=1.0):
    validate_same_shape(image, background, "Background image")
    return image - np.float32(scale) * np.asarray(background, dtype=np.float32)


def apply_monitor_normalization(image, monitor):
    if monitor <= 0:
        raise ValueError("Monitor / exposure normalization must be > 0.")
    return image / np.float32(monitor)


def robust_hot_pixel_mask(image, size=3, threshold_sigma=8.0):
    image = np.asarray(image, dtype=np.float32)
    fill = np.float32(np.nanmedian(image))
    work = np.nan_to_num(image, nan=fill)
    local = median_filter(work, size=max(1, int(size)), mode="nearest")
    resid = work - local
    # Sample robust statistics on large detectors; the local filter remains exact.
    step = max(1, int(np.sqrt(resid.size / 262144)))
    sample = resid[::step, ::step].ravel()
    med = np.median(sample)
    mad = np.median(np.abs(sample - med))
    sigma = 1.4826 * mad
    if sigma <= 0:
        sigma = np.std(sample)
    if sigma <= 0:
        return np.zeros_like(image, dtype=bool)
    return np.abs(resid - med) > threshold_sigma * sigma


def beamstop_mask(shape, center_x, center_y, radius):
    # ogrid avoids allocating two full coordinate arrays.
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius ** 2


def _sample_for_percentile(image, target=262144):
    arr = np.asarray(image)
    target = max(4096, int(target))
    step = max(1, int(np.sqrt(arr.size / target)))
    sample = arr[::step, ::step].ravel()
    return sample[np.isfinite(sample)]


def percentile_bounds(image, low, high, fast=True, target=262144):
    if fast:
        finite = _sample_for_percentile(image, target)
    else:
        arr = np.asarray(image)
        finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float(lo if np.isfinite(lo) else 0.0), float((lo + 1.0) if np.isfinite(lo) else 1.0)
    return float(lo), float(hi)



def display_level_bounds(image, settings: ProcessingSettings):
    """Return black/white levels for non-legacy contrast modes."""
    src = np.asarray(image, dtype=np.float32)
    mode = settings.contrast_mode.lower().strip()

    if mode == "manual":
        lo = float(settings.manual_black)
        hi = float(settings.manual_white)
    elif mode == "full range":
        finite = _sample_for_percentile(src, settings.percentile_sample_target) if settings.fast_percentiles else src[np.isfinite(src)]
        if finite.size == 0:
            return 0.0, 1.0
        lo = float(np.min(finite)); hi = float(np.max(finite))
    elif mode == "robust mad":
        finite = _sample_for_percentile(src, settings.percentile_sample_target) if settings.fast_percentiles else src[np.isfinite(src)]
        if finite.size == 0:
            return 0.0, 1.0
        med = float(np.median(finite))
        mad = float(np.median(np.abs(finite - med)))
        sigma = 1.4826 * mad
        if sigma <= 0:
            sigma = float(np.std(finite))
        if sigma <= 0:
            return med, med + 1.0
        # Slightly asymmetric limits are useful for positive diffraction peaks.
        lo = med - 2.0 * sigma
        hi = med + max(2.0, float(settings.robust_sigma)) * sigma
    elif mode == "source percentile":
        lo, hi = percentile_bounds(
            src, settings.percentile_low, settings.percentile_high,
            fast=settings.fast_percentiles,
            target=settings.percentile_sample_target,
        )
    else:
        # Legacy mode performs percentile clipping after the nonlinear tone curve.
        finite = src[np.isfinite(src)]
        if finite.size == 0:
            return 0.0, 1.0
        lo = float(np.min(finite)); hi = float(np.max(finite))

    if not np.isfinite(lo): lo = 0.0
    if not np.isfinite(hi) or hi <= lo: hi = lo + 1.0
    return float(lo), float(hi)



def parse_custom_curve_spec(spec):
    """Parse a semicolon-delimited 0..1 tone curve specification.

    Format: ``x,y;x,y;...``.  Invalid points are ignored.  Endpoints at
    x=0 and x=1 are added if missing.  Duplicate x values keep the last y.
    """
    points = []
    text = str(spec or '').strip()
    for token in text.split(';'):
        token = token.strip()
        if not token:
            continue
        try:
            xs, ys = token.split(',', 1)
            x = float(xs.strip())
            y = float(ys.strip())
        except Exception:
            continue
        if np.isfinite(x) and np.isfinite(y):
            points.append((float(np.clip(x, 0.0, 1.0)), float(np.clip(y, 0.0, 1.0))))

    if not points:
        return np.asarray([0.0, 1.0], dtype=np.float32), np.asarray([0.0, 1.0], dtype=np.float32)

    # Sort and de-duplicate by x, keeping the most recently supplied y.
    by_x = {}
    for x, y in points:
        by_x[round(x, 7)] = (x, y)
    points = sorted(by_x.values(), key=lambda p: p[0])

    if points[0][0] > 1e-7:
        points.insert(0, (0.0, points[0][1]))
    else:
        points[0] = (0.0, points[0][1])

    if points[-1][0] < 1.0 - 1e-7:
        points.append((1.0, points[-1][1]))
    else:
        points[-1] = (1.0, points[-1][1])

    xp = np.asarray([p[0] for p in points], dtype=np.float32)
    yp = np.asarray([p[1] for p in points], dtype=np.float32)
    return xp, yp


def apply_custom_tone_curve(image01, spec):
    """Apply a user-defined piecewise-linear curve to a normalized image.

    This is a display-only mapping from input intensity 0..1 to output
    brightness 0..1.  The quantitative diffraction array is not modified.
    """
    out = np.asarray(image01, dtype=np.float32)
    xp, yp = parse_custom_curve_spec(spec)
    if xp.size < 2:
        return out.copy()
    mapped = np.interp(out.ravel(), xp, yp).reshape(out.shape)
    return mapped.astype(np.float32, copy=False)


def _histogram_equalize_01(out, finite, bins):
    bins = max(128, int(bins))
    sample = out[finite]
    if sample.size < 2:
        return out
    hist, edges = np.histogram(sample, bins=bins, range=(0.0, 1.0))
    cdf = np.cumsum(hist, dtype=np.float64)
    if cdf[-1] <= 0:
        return out
    cdf /= cdf[-1]
    centers = 0.5 * (edges[:-1] + edges[1:])
    mapped = np.interp(out.ravel(), centers, cdf, left=0.0, right=1.0)
    return mapped.reshape(out.shape).astype(np.float32)


def _adaptive_local_contrast(out, finite, settings):
    strength = float(np.clip(settings.local_contrast_strength, 0.0, 1.0))
    if strength <= 0:
        return out

    sigma = max(1.0, float(settings.local_contrast_sigma))
    noise_floor_factor = max(0.0, float(settings.local_contrast_noise_floor))
    backend = resolve_backend(settings.compute_backend)

    if backend == "gpu":
        import cupy as cp
        from cupyx.scipy.ndimage import gaussian_filter as gpu_gaussian_filter
        x = cp.asarray(out, dtype=cp.float32)
        mean = gpu_gaussian_filter(x, sigma=sigma, mode="nearest")
        sqmean = gpu_gaussian_filter(x * x, sigma=sigma, mode="nearest")
        std = cp.sqrt(cp.maximum(sqmean - mean * mean, cp.float32(0.0)))
        finite_std = std[cp.asarray(finite)]
        med_std = cp.median(finite_std) if finite_std.size else cp.float32(0.0)
        floor = cp.maximum(cp.float32(1e-4), cp.float32(noise_floor_factor) * med_std)
        z = (x - mean) / (std + floor)
        local = cp.clip(cp.float32(0.5) + cp.float32(0.18) * z, 0.0, 1.0)
        mixed = (cp.float32(1.0 - strength) * x + cp.float32(strength) * local)
        result = cp.asnumpy(mixed).astype(np.float32, copy=False)
    else:
        x = np.asarray(out, dtype=np.float32)
        mean = gaussian_filter(x, sigma=sigma, mode="nearest", output=np.float32)
        sqmean = gaussian_filter(x * x, sigma=sigma, mode="nearest", output=np.float32)
        std = np.sqrt(np.maximum(sqmean - mean * mean, 0.0), dtype=np.float32)
        finite_std = std[finite]
        med_std = float(np.median(finite_std)) if finite_std.size else 0.0
        floor = max(1e-4, noise_floor_factor * med_std)
        z = (x - mean) / np.float32(std + floor)
        local = np.clip(np.float32(0.5) + np.float32(0.18) * z, 0.0, 1.0)
        result = ((1.0 - strength) * x + strength * local).astype(np.float32, copy=False)

    result[~finite] = 0.0
    return result


def display_transform(image, settings: ProcessingSettings):
    """Display-only contrast pipeline. Never modifies quantitative intensities."""
    src = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(src)
    if not np.any(finite):
        return np.zeros_like(src, dtype=np.float32)

    contrast_mode = settings.contrast_mode.lower().strip()

    if contrast_mode == "legacy percentile":
        # Preserve the v0.4 behavior exactly as the default.
        out = src.copy()
        minimum = np.nanmin(out)
        out -= np.float32(minimum)
        out[~finite] = 0.0
        maximum = np.max(out)
        if maximum > 0:
            out /= np.float32(maximum)
    else:
        lo, hi = display_level_bounds(src, settings)
        out = src.copy()
        out[~finite] = np.float32(lo)
        np.clip(out, lo, hi, out=out)
        out -= np.float32(lo)
        out /= np.float32(max(hi - lo, 1e-12))

    mode = settings.display_mode.lower().strip()
    if mode == "log":
        gain = np.float32(max(float(settings.log_gain), 1e-12))
        np.multiply(out, gain, out=out)
        np.log1p(out, out=out)
        out /= np.float32(np.log1p(gain))
    elif mode == "sqrt":
        np.sqrt(out, out=out)
    elif mode == "gamma":
        gamma = np.float32(max(float(settings.gamma), 1e-12))
        np.power(out, gamma, out=out)
    elif mode == "asinh":
        strength = np.float32(max(float(settings.asinh_strength), 1e-6))
        np.multiply(out, strength, out=out)
        np.arcsinh(out, out=out)
        out /= np.float32(np.arcsinh(strength))
    elif mode == "hist-eq":
        out = _histogram_equalize_01(out, finite, settings.histogram_bins)
    elif mode == "custom curve":
        out = apply_custom_tone_curve(out, settings.custom_curve_spec)
    elif mode != "linear":
        raise ValueError(f"Unknown display mode: {settings.display_mode}")

    if contrast_mode == "legacy percentile":
        lo, hi = percentile_bounds(
            out,
            settings.percentile_low,
            settings.percentile_high,
            fast=settings.fast_percentiles,
            target=settings.percentile_sample_target,
        )
        np.clip(out, lo, hi, out=out)
        out -= np.float32(lo)
        denom = hi - lo
        if denom > 0:
            out /= np.float32(denom)

    if settings.local_contrast_enabled:
        out = _adaptive_local_contrast(out, finite, settings)

    if settings.invert_display:
        out = np.float32(1.0) - out

    out[~finite] = 0.0
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

def _fast_gaussian_background_cpu(work, sigma, downsample):
    """Approximate a broad Gaussian background on a reduced grid."""
    factor = max(1, int(downsample))
    sigma = float(sigma)
    if factor <= 1 or sigma < 8.0:
        return gaussian_filter(work, sigma=max(sigma, 0.01), mode="nearest", output=np.float32)
    small = work[::factor, ::factor]
    bg_small = gaussian_filter(
        small,
        sigma=max(sigma / factor, 0.01),
        mode="nearest",
        output=np.float32,
    )
    factors = (work.shape[0] / bg_small.shape[0], work.shape[1] / bg_small.shape[1])
    bg = zoom(bg_small, factors, order=1, mode="nearest", prefilter=False)
    return bg[:work.shape[0], :work.shape[1]].astype(np.float32, copy=False)


def _fast_median_background_cpu(work, size, downsample):
    """Approximate a broad median background on a reduced grid, then interpolate."""
    factor = max(1, int(downsample))
    if factor <= 1 or size <= 9:
        return median_filter(work, size=size, mode="nearest")
    small = work[::factor, ::factor]
    small_size = max(3, int(round(size / factor)))
    if small_size % 2 == 0:
        small_size += 1
    bg_small = median_filter(small, size=small_size, mode="nearest")
    factors = (work.shape[0] / bg_small.shape[0], work.shape[1] / bg_small.shape[1])
    bg = zoom(bg_small, factors, order=1, mode="nearest", prefilter=False)
    return bg[:work.shape[0], :work.shape[1]].astype(np.float32, copy=False)


def process_quantitative_cpu(raw, settings, dark=None, flat=None, background=None):
    corrected = _as_float32(raw).copy()
    mask = ~np.isfinite(corrected)

    if settings.dark_enabled:
        if dark is None:
            raise ValueError("Dark correction enabled but no dark TIFF is loaded.")
        corrected = apply_dark(corrected, dark)

    if settings.flat_enabled:
        if flat is None:
            raise ValueError("Flat correction enabled but no flat TIFF is loaded.")
        corrected, flat_bad = apply_flat(corrected, flat)
        mask |= flat_bad

    if settings.background_enabled:
        if background is None:
            raise ValueError("Background subtraction enabled but no background TIFF is loaded.")
        corrected = apply_background(corrected, background, settings.background_scale)

    if settings.normalize_enabled:
        corrected = apply_monitor_normalization(corrected, settings.monitor_value)

    if settings.hot_pixels_enabled:
        mask |= robust_hot_pixel_mask(
            corrected,
            size=settings.hot_pixel_size,
            threshold_sigma=settings.hot_pixel_sigma,
        )

    if settings.saturation_enabled:
        mask |= corrected >= settings.saturation_value

    if settings.beamstop_enabled and settings.beamstop_radius > 0:
        mask |= beamstop_mask(
            corrected.shape,
            settings.beamstop_center_x,
            settings.beamstop_center_y,
            settings.beamstop_radius,
        )

    corrected_masked = corrected.copy()
    corrected_masked[mask] = np.nan
    return corrected_masked.astype(np.float32, copy=False), mask


def process_enhancement_cpu(corrected_masked, mask, settings):
    fill = np.nanmedian(corrected_masked)
    if not np.isfinite(fill):
        fill = 0.0
    work = np.nan_to_num(corrected_masked, nan=np.float32(fill)).astype(np.float32, copy=False)

    if settings.median_filter_enabled:
        work = median_filter(work, size=max(1, int(settings.median_filter_size)), mode="nearest")

    if settings.gaussian_filter_enabled:
        work = gaussian_filter(
            work,
            sigma=max(float(settings.gaussian_filter_sigma), 0.0),
            mode="nearest",
            output=np.float32,
        )

    background_model = None

    if settings.gaussian_background_enabled:
        if settings.fast_gaussian_background:
            background_model = _fast_gaussian_background_cpu(
                work,
                settings.gaussian_background_sigma,
                settings.gaussian_background_downsample,
            )
        else:
            background_model = gaussian_filter(
                work,
                sigma=max(float(settings.gaussian_background_sigma), 0.01),
                mode="nearest",
                output=np.float32,
            )
        work = work - background_model

    if settings.median_background_enabled:
        size = max(3, int(settings.median_background_size))
        if size % 2 == 0:
            size += 1
        if settings.fast_median_background:
            background_model = _fast_median_background_cpu(
                work, size, settings.median_background_downsample
            )
        else:
            background_model = median_filter(work, size=size, mode="nearest")
        work = work - background_model

    if settings.high_pass_enabled:
        low = gaussian_filter(
            work,
            sigma=max(float(settings.high_pass_sigma), 0.01),
            mode="nearest",
            output=np.float32,
        )
        work = work - low

    if settings.unsharp_enabled:
        blur = gaussian_filter(
            work,
            sigma=max(float(settings.unsharp_sigma), 0.01),
            mode="nearest",
            output=np.float32,
        )
        work = work + np.float32(settings.unsharp_amount) * (work - blur)

    enhanced = np.asarray(work, dtype=np.float32)
    enhanced[mask] = np.nan
    if background_model is not None:
        background_model = np.asarray(background_model, dtype=np.float32)
        background_model[mask] = np.nan
    return enhanced, background_model


def _gpu_modules():
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter as gpu_gaussian_filter
    from cupyx.scipy.ndimage import median_filter as gpu_median_filter
    from cupyx.scipy.ndimage import zoom as gpu_zoom
    return cp, gpu_gaussian_filter, gpu_median_filter, gpu_zoom


def process_quantitative_gpu(raw, settings, dark=None, flat=None, background=None):
    cp, _, gpu_median_filter, _ = _gpu_modules()
    corrected = cp.asarray(raw, dtype=cp.float32)
    mask = ~cp.isfinite(corrected)

    if settings.dark_enabled:
        if dark is None:
            raise ValueError("Dark correction enabled but no dark TIFF is loaded.")
        corrected = corrected - cp.asarray(dark, dtype=cp.float32)

    if settings.flat_enabled:
        if flat is None:
            raise ValueError("Flat correction enabled but no flat TIFF is loaded.")
        flat_g = cp.asarray(flat, dtype=cp.float32)
        valid = cp.isfinite(flat_g) & (flat_g > 1e-12)
        if not bool(cp.any(valid).get()):
            raise ValueError("Flat image contains no valid positive pixels.")
        norm = cp.nanmedian(flat_g[valid])
        bad = (~cp.isfinite(flat_g)) | (flat_g <= 1e-12)
        corrected = cp.where(bad, cp.nan, corrected * norm / flat_g)
        mask |= bad

    if settings.background_enabled:
        if background is None:
            raise ValueError("Background subtraction enabled but no background TIFF is loaded.")
        corrected = corrected - cp.float32(settings.background_scale) * cp.asarray(background, dtype=cp.float32)

    if settings.normalize_enabled:
        if settings.monitor_value <= 0:
            raise ValueError("Monitor / exposure normalization must be > 0.")
        corrected = corrected / cp.float32(settings.monitor_value)

    if settings.hot_pixels_enabled:
        fill = cp.nanmedian(corrected)
        work = cp.nan_to_num(corrected, nan=fill)
        local = gpu_median_filter(work, size=max(1, int(settings.hot_pixel_size)), mode="nearest")
        resid = work - local
        step = max(1, int(np.sqrt(corrected.size / 262144)))
        sample = resid[::step, ::step].ravel()
        med = cp.median(sample)
        mad = cp.median(cp.abs(sample - med))
        sigma = cp.float32(1.4826) * mad
        if float(sigma.get()) <= 0:
            sigma = cp.std(sample)
        if float(sigma.get()) > 0:
            mask |= cp.abs(resid - med) > cp.float32(settings.hot_pixel_sigma) * sigma

    if settings.saturation_enabled:
        mask |= corrected >= settings.saturation_value

    if settings.beamstop_enabled and settings.beamstop_radius > 0:
        yy = cp.arange(corrected.shape[0], dtype=cp.float32)[:, None]
        xx = cp.arange(corrected.shape[1], dtype=cp.float32)[None, :]
        mask |= (
            (xx - cp.float32(settings.beamstop_center_x)) ** 2
            + (yy - cp.float32(settings.beamstop_center_y)) ** 2
            <= cp.float32(settings.beamstop_radius) ** 2
        )

    corrected = cp.where(mask, cp.nan, corrected)
    return cp.asnumpy(corrected).astype(np.float32, copy=False), cp.asnumpy(mask)


def process_enhancement_gpu(corrected_masked, mask, settings):
    cp, gpu_gaussian_filter, gpu_median_filter, gpu_zoom = _gpu_modules()
    corrected = cp.asarray(corrected_masked, dtype=cp.float32)
    mask_g = cp.asarray(mask, dtype=cp.bool_)
    fill = cp.nanmedian(corrected)
    if not bool(cp.isfinite(fill).get()):
        fill = cp.float32(0.0)
    work = cp.nan_to_num(corrected, nan=fill)

    if settings.median_filter_enabled:
        work = gpu_median_filter(work, size=max(1, int(settings.median_filter_size)), mode="nearest")

    if settings.gaussian_filter_enabled:
        work = gpu_gaussian_filter(work, sigma=max(float(settings.gaussian_filter_sigma), 0.0), mode="nearest")

    background_model = None

    if settings.gaussian_background_enabled:
        sigma = max(float(settings.gaussian_background_sigma), 0.01)
        factor = max(1, int(settings.gaussian_background_downsample))
        if settings.fast_gaussian_background and factor > 1 and sigma >= 8.0:
            small = work[::factor, ::factor]
            bg_small = gpu_gaussian_filter(
                small, sigma=max(sigma / factor, 0.01), mode="nearest"
            )
            factors = (work.shape[0] / bg_small.shape[0], work.shape[1] / bg_small.shape[1])
            background_model = gpu_zoom(
                bg_small, factors, order=1, mode="nearest", prefilter=False
            )[:work.shape[0], :work.shape[1]]
        else:
            background_model = gpu_gaussian_filter(work, sigma=sigma, mode="nearest")
        work = work - background_model

    if settings.median_background_enabled:
        size = max(3, int(settings.median_background_size))
        if size % 2 == 0:
            size += 1
        factor = max(1, int(settings.median_background_downsample))
        if settings.fast_median_background and factor > 1 and size > 9:
            small = work[::factor, ::factor]
            small_size = max(3, int(round(size / factor)))
            if small_size % 2 == 0:
                small_size += 1
            bg_small = gpu_median_filter(small, size=small_size, mode="nearest")
            factors = (work.shape[0] / bg_small.shape[0], work.shape[1] / bg_small.shape[1])
            background_model = gpu_zoom(bg_small, factors, order=1, mode="nearest", prefilter=False)
            background_model = background_model[:work.shape[0], :work.shape[1]]
        else:
            background_model = gpu_median_filter(work, size=size, mode="nearest")
        work = work - background_model

    if settings.high_pass_enabled:
        low = gpu_gaussian_filter(work, sigma=max(float(settings.high_pass_sigma), 0.01), mode="nearest")
        work = work - low

    if settings.unsharp_enabled:
        blur = gpu_gaussian_filter(work, sigma=max(float(settings.unsharp_sigma), 0.01), mode="nearest")
        work = work + cp.float32(settings.unsharp_amount) * (work - blur)

    work = cp.where(mask_g, cp.nan, work)
    enhanced = cp.asnumpy(work).astype(np.float32, copy=False)
    if background_model is not None:
        background_model = cp.where(mask_g, cp.nan, background_model)
        background_model = cp.asnumpy(background_model).astype(np.float32, copy=False)
    return enhanced, background_model


def process_quantitative(raw, settings, dark=None, flat=None, background=None):
    backend = resolve_backend(settings.compute_backend)
    if backend == "gpu":
        return process_quantitative_gpu(raw, settings, dark, flat, background)
    return process_quantitative_cpu(raw, settings, dark, flat, background)


def process_enhancement(corrected, mask, settings):
    backend = resolve_backend(settings.compute_backend)
    if backend == "gpu":
        return process_enhancement_gpu(corrected, mask, settings)
    return process_enhancement_cpu(corrected, mask, settings)


class ProcessingEngine:
    """Stage cache so display changes do not rerun detector corrections or heavy filters."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.quant_key = None
        self.enhance_key = None
        self.display_key = None
        self.corrected = None
        self.mask = None
        self.enhanced = None
        self.background_model = None
        self.display = None

    def process(self, raw, settings, dark=None, flat=None, background=None):
        quant_key = (
            id(raw), id(dark), id(flat), id(background), raw.shape,
            _signature(settings, QUANTITATIVE_FIELDS),
        )
        stage = "cached"

        if quant_key != self.quant_key:
            self.corrected, self.mask = process_quantitative(
                raw, settings, dark=dark, flat=flat, background=background
            )
            self.quant_key = quant_key
            self.enhance_key = None
            self.display_key = None
            stage = "quantitative"

        enhance_key = (self.quant_key, _signature(settings, ENHANCEMENT_FIELDS))
        if enhance_key != self.enhance_key:
            self.enhanced, self.background_model = process_enhancement(
                self.corrected, self.mask, settings
            )
            self.enhance_key = enhance_key
            self.display_key = None
            if stage == "cached":
                stage = "enhancement"

        display_key = (self.enhance_key, _signature(settings, DISPLAY_FIELDS))
        if display_key != self.display_key:
            self.display = display_transform(self.enhanced, settings)
            self.display[self.mask] = 0.0
            self.display_key = display_key
            if stage == "cached":
                stage = "display"

        return {
            "raw": _as_float32(raw),
            "corrected": self.corrected,
            "enhanced": self.enhanced,
            "display": self.display,
            "mask": self.mask,
            "background_model": self.background_model,
            "_recomputed_stage": stage,
        }


def process_image(raw, settings, dark=None, flat=None, background=None):
    engine = ProcessingEngine()
    return engine.process(raw, settings, dark=dark, flat=flat, background=background)
