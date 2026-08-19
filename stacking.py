from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import os
import gc
from collections import Counter

import numpy as np
from scipy.fft import rfft2, irfft2

from .backend import resolve_backend
from .io_utils import load_tiff, default_tiff_workers, inspect_tiff_output_shape


@dataclass
class StackSettings:
    method: str = "mean"
    align: bool = False
    max_shift_pixels: int = 12
    sigma_clip_threshold: float = 4.0
    sigma_clip_iterations: int = 2
    trim_fraction: float = 0.10
    winsor_fraction: float = 0.05
    huber_delta: float = 1.5
    huber_iterations: int = 3
    noise_weight_floor: float = 1e-6
    chunk_rows: int = 128

    # Performance
    compute_backend: str = "auto"
    fft_workers: int = -1
    registration_crop_size: int = 1536
    tiff_workers: int = 0  # 0 = automatic

    def to_dict(self):
        return asdict(self)


@dataclass
class StackShapePreflight:
    expected_shape: tuple[int, int]
    compatible_paths: list[Path]
    incompatible: list[tuple[Path, tuple[int, int]]]
    shapes: dict[str, tuple[int, int]]


def preflight_stack_shapes(paths) -> StackShapePreflight:
    """Inspect TIFF dimensions before allocating stack buffers.

    The most common output shape is chosen as the expected shape.  This makes
    a single thumbnail / processed TIFF in the same folder an outlier even if
    it happens to be the first selected file.
    """
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError('No TIFF files were selected for stacking.')

    shape_pairs = []
    shape_map = {}
    errors = []
    for path in paths:
        try:
            shape = tuple(inspect_tiff_output_shape(path))
            shape_pairs.append((path, shape))
            shape_map[str(path)] = shape
        except Exception as exc:
            errors.append((path, str(exc)))

    if errors:
        lines = '\n'.join(f'  • {p.name}: {msg}' for p, msg in errors[:12])
        more = '' if len(errors) <= 12 else f'\n  … and {len(errors)-12} more.'
        raise ValueError('One or more TIFF files could not be inspected:\n' + lines + more)

    counts = Counter(shape for _, shape in shape_pairs)
    expected_shape = counts.most_common(1)[0][0]
    compatible = [p for p, shape in shape_pairs if shape == expected_shape]
    incompatible = [(p, shape) for p, shape in shape_pairs if shape != expected_shape]

    return StackShapePreflight(
        expected_shape=expected_shape,
        compatible_paths=compatible,
        incompatible=incompatible,
        shapes=shape_map,
    )


def _shape_mismatch_message(preflight: StackShapePreflight) -> str:
    h, w = preflight.expected_shape
    lines = [
        f'Included TIFFs do not all have the same dimensions.',
        f'Most common / expected image size: {w} × {h} pixels (array shape {preflight.expected_shape}).',
        '',
        'Incompatible file(s):',
    ]
    for path, shape in preflight.incompatible[:15]:
        ih, iw = shape
        lines.append(f'  • {path.name}: {iw} × {ih} pixels (shape {shape})')
    if len(preflight.incompatible) > 15:
        lines.append(f'  … and {len(preflight.incompatible)-15} more.')
    lines.extend([
        '',
        'Exclude these files or place different detector/image sizes in separate stacks.'
    ])
    return '\n'.join(lines)



@dataclass
class FrameStat:
    path: str
    filename: str
    included: bool = True
    mean: float | None = None
    median: float | None = None
    maximum: float | None = None
    correlation_to_reference: float | None = None
    noise_sigma: float | None = None
    stack_weight: float | None = None
    shift_y: int = 0
    shift_x: int = 0

    def to_dict(self):
        return asdict(self)


@dataclass
class StackResult:
    image: np.ndarray
    frame_stats: list[FrameStat]
    settings: StackSettings
    shape: tuple[int, int]

    def summary(self):
        n = len(self.frame_stats)
        method = self.settings.method.lower().strip()
        weights = np.asarray([
            stat.stack_weight if stat.stack_weight is not None else 1.0
            for stat in self.frame_stats
        ], dtype=float)
        effective_n = None
        if weights.size and np.sum(weights * weights) > 0:
            effective_n = float((np.sum(weights) ** 2) / np.sum(weights * weights))
        snr_gain = None
        if n > 0 and method in {
            "mean", "sum", "inverse-variance weighted mean"
        }:
            snr_gain = float(np.sqrt(effective_n if method == "inverse-variance weighted mean" else n))
        return {
            "frame_count": n,
            "method": self.settings.method,
            "aligned": self.settings.align,
            "shape": list(self.shape),
            "compute_backend": resolve_backend(self.settings.compute_backend),
            "effective_frame_count": effective_n,
            "approximate_snr_gain_vs_one_frame": snr_gain,
        }


def _io_workers(settings):
    return int(settings.tiff_workers) if int(settings.tiff_workers) > 0 else default_tiff_workers()


def _sampled_stats(image, target=262144):
    image = np.asarray(image)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return None, None, None
    mean = float(np.mean(finite, dtype=np.float64))
    maximum = float(np.max(finite))
    step = max(1, int(np.sqrt(image.size / max(4096, target))))
    sample = image[::step, ::step].ravel()
    sample = sample[np.isfinite(sample)]
    median = float(np.median(sample)) if sample.size else None
    return mean, median, maximum


def _center_crop(image, max_size):
    h, w = image.shape
    max_size = max(64, int(max_size))
    ch, cw = min(h, max_size), min(w, max_size)
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    return image[y0:y0+ch, x0:x0+cw]


def _registration_image(image, crop_size=1536):
    work = _center_crop(np.asarray(image, dtype=np.float32), crop_size)
    finite = np.isfinite(work)
    fill = np.float32(np.nanmedian(work[finite]) if np.any(finite) else 0.0)
    work = np.nan_to_num(work, nan=fill).astype(np.float32, copy=False)
    work = work - np.float32(np.mean(work, dtype=np.float64))
    scale = np.float32(np.std(work, dtype=np.float64))
    if scale > 0:
        work = work / scale
    return work


@dataclass
class RegistrationReference:
    shape: tuple[int, int]
    spectrum: object
    reference_image: np.ndarray
    backend: str
    workers: int
    crop_size: int
    max_shift_pixels: int


def prepare_registration_reference(reference, settings):
    backend = resolve_backend(settings.compute_backend)
    ref = _registration_image(reference, settings.registration_crop_size)
    if backend == "gpu":
        import cupy as cp
        spectrum = cp.fft.rfft2(cp.asarray(ref, dtype=cp.float32))
    else:
        spectrum = rfft2(ref, workers=settings.fft_workers)
    return RegistrationReference(
        shape=ref.shape,
        spectrum=spectrum,
        reference_image=ref,
        backend=backend,
        workers=settings.fft_workers,
        crop_size=settings.registration_crop_size,
        max_shift_pixels=int(settings.max_shift_pixels),
    )


def _overlap_for_shift(reference, moving, dy, dx, stride=1):
    """Return sampled overlapping views when `moving` is shifted by (dy, dx)."""
    h, w = reference.shape
    if dy >= 0:
        ry0, ry1 = dy, h
        my0, my1 = 0, h - dy
    else:
        ry0, ry1 = 0, h + dy
        my0, my1 = -dy, h

    if dx >= 0:
        rx0, rx1 = dx, w
        mx0, mx1 = 0, w - dx
    else:
        rx0, rx1 = 0, w + dx
        mx0, mx1 = -dx, w

    return (
        reference[ry0:ry1:stride, rx0:rx1:stride],
        moving[my0:my1:stride, mx0:mx1:stride],
    )


def _shift_score(reference, moving, dy, dx, stride):
    a, b = _overlap_for_shift(reference, moving, dy, dx, stride)
    if a.size < 100:
        return -np.inf
    # Inputs are already globally centered/scaled. Re-centering the sampled
    # overlap makes the score more robust to background changes near edges.
    aa = a - np.mean(a, dtype=np.float64)
    bb = b - np.mean(b, dtype=np.float64)
    denom = np.sqrt(
        np.sum(aa * aa, dtype=np.float64)
        * np.sum(bb * bb, dtype=np.float64)
    )
    if denom <= 0:
        return -np.inf
    return float(np.sum(aa * bb, dtype=np.float64) / denom)


def _small_shift_search(reference, moving, max_shift):
    """Robust two-stage integer registration for the small drifts typical of CCD series."""
    max_shift = max(0, int(max_shift))
    # Coarse grid: use a sparse sample of the registration crop.
    coarse_stride = max(2, int(np.ceil(max(reference.shape) / 256)))
    best = (0, 0)
    best_score = -np.inf
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            score = _shift_score(reference, moving, dy, dx, coarse_stride)
            if score > best_score:
                best_score = score
                best = (dy, dx)

    # Fine refinement in a small neighborhood at denser sampling.
    fine_stride = max(1, coarse_stride // 2)
    by, bx = best
    best_score = -np.inf
    refined = best
    for dy in range(max(-max_shift, by - 2), min(max_shift, by + 2) + 1):
        for dx in range(max(-max_shift, bx - 2), min(max_shift, bx + 2) + 1):
            score = _shift_score(reference, moving, dy, dx, fine_stride)
            if score > best_score:
                best_score = score
                refined = (dy, dx)
    return refined


def estimate_integer_translation(reference_info, moving):
    """Estimate integer shift using cached FFT cross-correlation.

    The registration images are centered and variance-normalized first, so a
    normal cross-correlation is robust to overall intensity scaling. This is
    faster and more stable for sparse diffraction patterns than phase-only
    correlation.
    """
    mov = _registration_image(moving, reference_info.crop_size)
    if mov.shape != reference_info.shape:
        raise ValueError("Registration crop shape changed between frames.")

    if reference_info.backend == "gpu":
        import cupy as cp
        fm = cp.fft.rfft2(cp.asarray(mov, dtype=cp.float32))
        corr = cp.fft.irfft2(
            reference_info.spectrum * cp.conj(fm),
            s=reference_info.shape,
        )
        peak_flat = int(cp.argmax(corr).get())
        peak = np.unravel_index(peak_flat, reference_info.shape)
    else:
        fm = rfft2(mov, workers=reference_info.workers)
        corr = irfft2(
            reference_info.spectrum * np.conj(fm),
            s=reference_info.shape,
            workers=reference_info.workers,
        )
        peak = np.unravel_index(np.argmax(corr), reference_info.shape)

    shift = np.array(peak, dtype=int)
    for dim, n in enumerate(reference_info.shape):
        if shift[dim] > n // 2:
            shift[dim] -= n
    return int(shift[0]), int(shift[1])


def apply_integer_translation(image, dy, dx):
    image = np.asarray(image, dtype=np.float32)
    out = np.roll(image, shift=(dy, dx), axis=(0, 1)).copy()
    if dy > 0:
        out[:dy, :] = np.nan
    elif dy < 0:
        out[dy:, :] = np.nan
    if dx > 0:
        out[:, :dx] = np.nan
    elif dx < 0:
        out[:, dx:] = np.nan
    return out


def sampled_correlation(reference, moving, target=262144):
    step = max(1, int(np.sqrt(reference.size / max(4096, target))))
    a = np.asarray(reference)[::step, ::step].ravel()
    b = np.asarray(moving)[::step, ::step].ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(valid) < 10:
        return np.nan
    a = a[valid]; b = b[valid]
    sa = np.std(a); sb = np.std(b)
    return float(np.corrcoef(a, b)[0, 1]) if sa > 0 and sb > 0 else np.nan


def _sampled_noise_sigma(image, target=131072):
    """Robust frame-level noise estimate from first differences.

    For comparable exposures this is useful for inverse-variance frame weights.
    It intentionally estimates high-frequency noise rather than total image variance.
    """
    arr = np.asarray(image, dtype=np.float32)
    step = max(1, int(np.sqrt(arr.size / max(4096, int(target)))))
    sample = arr[::step, ::step]
    dx = np.diff(sample, axis=1).ravel()
    dy = np.diff(sample, axis=0).ravel()
    diffs = np.concatenate([dx, dy])
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size < 100:
        return np.nan
    med = np.median(diffs)
    mad = np.median(np.abs(diffs - med))
    sigma = 1.4826 * mad / np.sqrt(2.0)
    if sigma <= 0:
        sigma = np.std(diffs) / np.sqrt(2.0)
    return float(sigma) if sigma > 0 else np.nan


def _load_and_prepare_frame(path, expected_shape, reference, settings, registration_reference=None):
    image, meta = load_tiff(path, dtype=np.float32, maxworkers=_io_workers(settings))
    if image.shape != expected_shape:
        raise ValueError(f"{Path(path).name} has shape {image.shape}; expected {expected_shape}.")

    dy = dx = 0
    if settings.align and registration_reference is not None:
        dy, dx = estimate_integer_translation(registration_reference, image)
        if abs(dy) > settings.max_shift_pixels or abs(dx) > settings.max_shift_pixels:
            raise ValueError(
                f"{Path(path).name}: estimated registration shift ({dy}, {dx}) "
                f"exceeds max shift {settings.max_shift_pixels} px."
            )
        image = apply_integer_translation(image, dy, dx)

    mean, median, maximum = _sampled_stats(image)
    corr = sampled_correlation(reference, image) if reference is not None else 1.0
    noise_sigma = _sampled_noise_sigma(image)
    stat = FrameStat(
        path=str(path), filename=Path(path).name, included=True,
        mean=mean, median=median, maximum=maximum,
        correlation_to_reference=corr, noise_sigma=noise_sigma, stack_weight=None,
        shift_y=dy, shift_x=dx,
    )
    all_finite = np.issubdtype(np.dtype(meta["dtype"]), np.integer) and not settings.align
    return image, stat, all_finite


def _sigma_clipped_mean_block_cpu(block, threshold, iterations):
    data = np.asarray(block, dtype=np.float32).copy()
    for _ in range(max(1, int(iterations))):
        med = np.nanmedian(data, axis=0).astype(np.float32)
        mad = np.nanmedian(np.abs(data - med[None, :, :]), axis=0).astype(np.float32)
        sigma = np.float32(1.4826) * mad
        std = np.nanstd(data, axis=0, dtype=np.float32)
        sigma = np.where(sigma > 0, sigma, std)
        good = np.abs(data - med[None, :, :]) <= np.float32(threshold) * sigma[None, :, :]
        good |= sigma[None, :, :] == 0
        data[~good] = np.nan
    return np.nanmean(data, axis=0, dtype=np.float32)



def _trimmed_mean_block_cpu(block, fraction):
    fraction = float(np.clip(fraction, 0.0, 0.45))
    data = np.asarray(block, dtype=np.float32)
    if fraction <= 0:
        return np.nanmean(data, axis=0, dtype=np.float32)
    lo = np.nanpercentile(data, 100.0 * fraction, axis=0)
    hi = np.nanpercentile(data, 100.0 * (1.0 - fraction), axis=0)
    keep = (data >= lo[None, :, :]) & (data <= hi[None, :, :])
    return np.nanmean(np.where(keep, data, np.nan), axis=0, dtype=np.float32)


def _winsorized_mean_block_cpu(block, fraction):
    fraction = float(np.clip(fraction, 0.0, 0.45))
    data = np.asarray(block, dtype=np.float32)
    if fraction <= 0:
        return np.nanmean(data, axis=0, dtype=np.float32)
    lo = np.nanpercentile(data, 100.0 * fraction, axis=0)
    hi = np.nanpercentile(data, 100.0 * (1.0 - fraction), axis=0)
    clipped = np.minimum(np.maximum(data, lo[None, :, :]), hi[None, :, :])
    clipped[~np.isfinite(data)] = np.nan
    return np.nanmean(clipped, axis=0, dtype=np.float32)


def _minmax_rejected_mean_block_cpu(block):
    data = np.asarray(block, dtype=np.float32)
    count = np.sum(np.isfinite(data), axis=0)
    total = np.nansum(data, axis=0, dtype=np.float64)
    with np.errstate(all='ignore'):
        low = np.nanmin(data, axis=0)
        high = np.nanmax(data, axis=0)
    out = np.full(count.shape, np.nan, dtype=np.float32)
    enough = count >= 4
    out[enough] = ((total[enough] - low[enough] - high[enough]) / (count[enough] - 2)).astype(np.float32)
    fallback = (count > 0) & (~enough)
    out[fallback] = (total[fallback] / count[fallback]).astype(np.float32)
    return out


def _huber_mean_block_cpu(block, delta, iterations):
    data = np.asarray(block, dtype=np.float32)
    center = np.nanmedian(data, axis=0).astype(np.float32)
    mad = np.nanmedian(np.abs(data - center[None, :, :]), axis=0).astype(np.float32)
    scale = np.float32(1.4826) * mad
    fallback = np.nanstd(data, axis=0, dtype=np.float32)
    scale = np.where(scale > 1e-8, scale, fallback)
    scale = np.where(scale > 1e-8, scale, np.float32(1.0))
    d = np.float32(max(0.1, float(delta)))

    for _ in range(max(1, int(iterations))):
        resid = (data - center[None, :, :]) / scale[None, :, :]
        absr = np.abs(resid)
        weights = np.where(absr <= d, 1.0, d / np.maximum(absr, 1e-12)).astype(np.float32)
        weights[~np.isfinite(data)] = 0.0
        denom = np.sum(weights, axis=0, dtype=np.float32)
        numer = np.nansum(weights * data, axis=0, dtype=np.float32)
        valid = denom > 0
        center = np.where(valid, numer / np.maximum(denom, 1e-12), np.nan).astype(np.float32)
    return center



def _cupy_nanquantile_axis0(data, q):
    """NaN-aware linear quantile along frame axis for CuPy arrays.

    CuPy exposes ``percentile`` / ``quantile`` and several NaN-aware summary
    functions, but not ``nanpercentile``.  For XRFD stack chunks the frame
    axis is normally small, so sorting only axis 0 is a practical and robust
    GPU implementation.
    """
    import cupy as cp

    q = float(np.clip(q, 0.0, 1.0))
    valid = cp.isfinite(data)
    count = cp.sum(valid, axis=0, dtype=cp.int32)

    # NaNs are placed after all finite samples.  Quantile indices are then
    # computed from the finite count independently at every detector pixel.
    ordered = cp.sort(
        cp.where(valid, data, cp.float32(cp.inf)),
        axis=0,
    )

    last = cp.maximum(count - 1, 0)
    position = cp.float32(q) * last.astype(cp.float32)
    i0 = cp.floor(position).astype(cp.int32)
    i1 = cp.ceil(position).astype(cp.int32)
    frac = position - i0.astype(cp.float32)

    v0 = cp.take_along_axis(ordered, i0[None, :, :], axis=0)[0]
    v1 = cp.take_along_axis(ordered, i1[None, :, :], axis=0)[0]
    out = v0 + frac * (v1 - v0)
    return cp.where(count > 0, out, cp.nan)


def _combine_block_cpu_dispatch(block, method, settings):
    """CPU fallback shared by normal CPU mode and GPU compatibility fallback."""
    if method == 'median':
        return np.nanmedian(block, axis=0).astype(np.float32)
    if method == 'sigma-clipped mean':
        return _sigma_clipped_mean_block_cpu(
            block, settings.sigma_clip_threshold, settings.sigma_clip_iterations
        )
    if method == 'trimmed mean':
        return _trimmed_mean_block_cpu(block, settings.trim_fraction)
    if method == 'winsorized mean':
        return _winsorized_mean_block_cpu(block, settings.winsor_fraction)
    if method == 'min/max rejected mean':
        return _minmax_rejected_mean_block_cpu(block)
    if method == 'huber mean':
        return _huber_mean_block_cpu(
            block, settings.huber_delta, settings.huber_iterations
        )
    raise ValueError(f'Unhandled stack method {method!r}.')


def _combine_block_gpu(block, method, settings):
    import cupy as cp
    data = cp.asarray(block, dtype=cp.float32)

    if method == 'median':
        out = cp.nanmedian(data, axis=0)
    elif method == 'sigma-clipped mean':
        for _ in range(max(1, int(settings.sigma_clip_iterations))):
            med = cp.nanmedian(data, axis=0)
            mad = cp.nanmedian(cp.abs(data - med[None, :, :]), axis=0)
            sigma = cp.float32(1.4826) * mad
            std = cp.nanstd(data, axis=0)
            sigma = cp.where(sigma > 0, sigma, std)
            good = cp.abs(data - med[None, :, :]) <= cp.float32(settings.sigma_clip_threshold) * sigma[None, :, :]
            good |= sigma[None, :, :] == 0
            data = cp.where(good, data, cp.nan)
        out = cp.nanmean(data, axis=0)
    elif method == 'trimmed mean':
        f = float(np.clip(settings.trim_fraction, 0.0, 0.45))
        lo = _cupy_nanquantile_axis0(data, f)
        hi = _cupy_nanquantile_axis0(data, 1.0 - f)
        keep = (data >= lo[None, :, :]) & (data <= hi[None, :, :])
        out = cp.nanmean(cp.where(keep, data, cp.nan), axis=0)
    elif method == 'winsorized mean':
        f = float(np.clip(settings.winsor_fraction, 0.0, 0.45))
        lo = _cupy_nanquantile_axis0(data, f)
        hi = _cupy_nanquantile_axis0(data, 1.0 - f)
        clipped = cp.minimum(cp.maximum(data, lo[None, :, :]), hi[None, :, :])
        clipped = cp.where(cp.isfinite(data), clipped, cp.nan)
        out = cp.nanmean(clipped, axis=0)
    elif method == 'min/max rejected mean':
        count = cp.sum(cp.isfinite(data), axis=0)
        total = cp.nansum(data, axis=0)
        low = cp.nanmin(data, axis=0)
        high = cp.nanmax(data, axis=0)
        enough = count >= 4
        normal = total / cp.maximum(count, 1)
        rejected = (total - low - high) / cp.maximum(count - 2, 1)
        out = cp.where(enough, rejected, normal)
        out = cp.where(count > 0, out, cp.nan)
    elif method == 'huber mean':
        center = cp.nanmedian(data, axis=0)
        mad = cp.nanmedian(cp.abs(data - center[None, :, :]), axis=0)
        scale = cp.float32(1.4826) * mad
        fallback = cp.nanstd(data, axis=0)
        scale = cp.where(scale > 1e-8, scale, fallback)
        scale = cp.where(scale > 1e-8, scale, cp.float32(1.0))
        d = cp.float32(max(0.1, float(settings.huber_delta)))
        for _ in range(max(1, int(settings.huber_iterations))):
            resid = (data - center[None, :, :]) / scale[None, :, :]
            absr = cp.abs(resid)
            weights = cp.where(absr <= d, cp.float32(1.0), d / cp.maximum(absr, cp.float32(1e-12)))
            weights = cp.where(cp.isfinite(data), weights, cp.float32(0.0))
            denom = cp.sum(weights, axis=0)
            numer = cp.nansum(weights * data, axis=0)
            center = cp.where(denom > 0, numer / cp.maximum(denom, cp.float32(1e-12)), cp.nan)
        out = center
    else:
        raise ValueError(f'GPU block combiner does not support {method!r}.')

    return cp.asnumpy(out).astype(np.float32, copy=False)


def build_stack(paths, settings: StackSettings, progress_callback=None):
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError('No TIFF files were selected for stacking.')

    method = settings.method.lower().strip()
    allowed = {
        'mean', 'sum', 'median', 'sigma-clipped mean',
        'trimmed mean', 'winsorized mean', 'min/max rejected mean',
        'inverse-variance weighted mean', 'huber mean',
    }
    if method not in allowed:
        raise ValueError(f'Unknown stack method {settings.method!r}.')

    preflight = preflight_stack_shapes(paths)
    if preflight.incompatible:
        raise ValueError(_shape_mismatch_message(preflight))

    expected_shape = preflight.expected_shape
    reference, _ = load_tiff(paths[0], dtype=np.float32, maxworkers=_io_workers(settings))
    if reference.shape != expected_shape:
        # Defensive guard: metadata inspection and decoded data should agree.
        raise ValueError(
            f'{paths[0].name} decoded to shape {reference.shape}; expected {expected_shape}.'
        )
    registration_reference = prepare_registration_reference(reference, settings) if settings.align else None
    frame_stats = []

    def report(stage, current, total, message=''):
        if progress_callback:
            progress_callback({'stage': stage, 'current': int(current), 'total': int(total), 'message': message})

    if method in {'mean', 'sum'}:
        accumulator = np.zeros(expected_shape, dtype=np.float64)
        counts = None
        scalar_count = 0

        for i, path in enumerate(paths, start=1):
            report('loading', i - 1, len(paths), f'Loading {path.name}')
            image, stat, all_finite = _load_and_prepare_frame(
                path, expected_shape, reference, settings, registration_reference
            )
            stat.stack_weight = 1.0
            frame_stats.append(stat)

            if all_finite and counts is None:
                accumulator += image
                scalar_count += 1
            else:
                if counts is None:
                    counts = np.full(expected_shape, scalar_count, dtype=np.uint16)
                valid = np.isfinite(image)
                accumulator[valid] += image[valid]
                counts[valid] += 1
            report('loading', i, len(paths), f'Added {path.name}')

        if counts is None:
            result = accumulator / max(1, scalar_count) if method == 'mean' else accumulator
        else:
            result = np.full(expected_shape, np.nan, dtype=np.float64)
            valid = counts > 0
            result[valid] = accumulator[valid] / counts[valid] if method == 'mean' else accumulator[valid]
        result = result.astype(np.float32)

    elif method == 'inverse-variance weighted mean':
        numerator = np.zeros(expected_shape, dtype=np.float64)
        denominator = np.zeros(expected_shape, dtype=np.float64)
        floor = max(float(settings.noise_weight_floor), 1e-12)

        for i, path in enumerate(paths, start=1):
            report('loading', i - 1, len(paths), f'Loading {path.name}')
            image, stat, _ = _load_and_prepare_frame(
                path, expected_shape, reference, settings, registration_reference
            )
            sigma = stat.noise_sigma
            weight = 1.0 / max((sigma * sigma) if sigma is not None and np.isfinite(sigma) else floor, floor)
            stat.stack_weight = float(weight)
            frame_stats.append(stat)
            valid = np.isfinite(image)
            numerator[valid] += weight * image[valid]
            denominator[valid] += weight
            report('loading', i, len(paths), f'Added {path.name}')

        # Rescale reported weights to mean 1 for readability; this does not alter result.
        weights = np.asarray([st.stack_weight for st in frame_stats], dtype=float)
        meanw = np.mean(weights) if weights.size else 1.0
        if meanw > 0:
            for st in frame_stats:
                st.stack_weight = float(st.stack_weight / meanw)

        result = np.full(expected_shape, np.nan, dtype=np.float32)
        good = denominator > 0
        result[good] = (numerator[good] / denominator[good]).astype(np.float32)

    else:
        n = len(paths); h, w = expected_shape
        backend = resolve_backend(settings.compute_backend)

        # Robust methods need access to several frames at each pixel.  Keep the
        # backing array on disk, but explicitly close its mmap handle before
        # TemporaryDirectory cleanup.  This is required on Windows, where an
        # open NumPy memmap prevents stack.npy from being deleted.
        with TemporaryDirectory(prefix='xrd_stack_') as tmp:
            mmap_path = Path(tmp) / 'stack.npy'
            mm = None
            block = None
            out = None
            try:
                mm = np.lib.format.open_memmap(
                    mmap_path,
                    mode='w+',
                    dtype=np.float32,
                    shape=(n, h, w),
                )

                for i, path in enumerate(paths):
                    report('loading', i, n, f'Loading {path.name}')
                    image, stat, _ = _load_and_prepare_frame(
                        path, expected_shape, reference, settings, registration_reference
                    )
                    stat.stack_weight = 1.0
                    frame_stats.append(stat)
                    mm[i] = image
                    report('loading', i + 1, n, f'Added {path.name}')

                mm.flush()

                result = np.full(expected_shape, np.nan, dtype=np.float32)
                chunk = max(1, int(settings.chunk_rows))
                total_blocks = (h + chunk - 1) // chunk

                for bi, y0 in enumerate(range(0, h, chunk), start=1):
                    y1 = min(h, y0 + chunk)
                    report('combining', bi - 1, total_blocks, f'Combining rows {y0}:{y1}')

                    # Force a RAM copy.  A plain np.asarray() can retain a view
                    # onto the memmap and therefore keep the file locked on Windows.
                    block = np.array(
                        mm[:, y0:y1, :],
                        dtype=np.float32,
                        copy=True,
                    )

                    if backend == 'gpu':
                        try:
                            out = _combine_block_gpu(block, method, settings)
                        except Exception as gpu_exc:
                            # Auto/GPU mode must never make a scientifically valid
                            # stack impossible merely because a CuPy API differs
                            # across installed versions.  Fall back for the current
                            # and remaining chunks and report the reason in the GUI.
                            report(
                                'combining', bi - 1, total_blocks,
                                f'GPU {method} unavailable ({type(gpu_exc).__name__}: {gpu_exc}); '
                                'falling back to CPU.'
                            )
                            backend = 'cpu'
                            out = _combine_block_cpu_dispatch(block, method, settings)
                    else:
                        out = _combine_block_cpu_dispatch(block, method, settings)

                    result[y0:y1] = out
                    report('combining', bi, total_blocks, f'Combined rows {y0}:{y1}')

                    # Release chunk references promptly.
                    block = None
                    out = None
            finally:
                block = None
                out = None
                if mm is not None:
                    try:
                        mm.flush()
                    except Exception:
                        pass
                    # Close the OS-level memory map explicitly before the
                    # TemporaryDirectory context tries to delete stack.npy.
                    mmap_handle = getattr(mm, '_mmap', None)
                    if mmap_handle is not None:
                        try:
                            mmap_handle.close()
                        except Exception:
                            pass
                    mm = None
                gc.collect()
    report('done', 1, 1, 'Stack complete')
    return StackResult(image=result, frame_stats=frame_stats, settings=settings, shape=expected_shape)
