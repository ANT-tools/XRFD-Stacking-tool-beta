from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
from scipy.fft import rfft2, irfft2
from scipy.ndimage import map_coordinates


@dataclass
class AxisDetectionSettings:
    analysis_max_dimension: int = 700
    center_search_radius_px: float = 30.0
    center_refine_radius_px: float = 3.0
    center_refine_step_px: float = 0.5
    coarse_angle_step_deg: float = 2.0
    fine_angle_step_deg: float = 0.1
    central_exclusion_radius_px: float = 20.0
    sample_target: int = 50000
    intensity_floor_percentile: float = 55.0

    def to_dict(self):
        return asdict(self)


@dataclass
class AxisDetectionResult:
    center_x: float
    center_y: float
    fiber_angle_deg: float
    center_symmetry_score: float
    mirror_symmetry_score: float
    coarse_angles_deg: np.ndarray
    coarse_scores: np.ndarray
    fine_angles_deg: np.ndarray
    fine_scores: np.ndarray
    downsample_step: int
    settings: AxisDetectionSettings

    def summary(self):
        return {
            'center_x_px': float(self.center_x),
            'center_y_px': float(self.center_y),
            'fiber_angle_deg': float(self.fiber_angle_deg),
            'center_symmetry_score': float(self.center_symmetry_score),
            'mirror_symmetry_score': float(self.mirror_symmetry_score),
            'downsample_step': int(self.downsample_step),
            'settings': self.settings.to_dict(),
        }


def _prepare_image(image, mask, settings):
    image = np.asarray(image, dtype=np.float32)
    h, w = image.shape
    step = max(1, int(np.ceil(max(h, w) / max(128, int(settings.analysis_max_dimension)))))
    small = image[::step, ::step].astype(np.float32, copy=False)

    if mask is None:
        valid = np.isfinite(small)
    else:
        valid = (~np.asarray(mask, dtype=bool)[::step, ::step]) & np.isfinite(small)

    values = small[valid]
    if values.size < 100:
        raise ValueError('Too few valid pixels for automatic center/axis detection.')

    p_low, p_high = np.percentile(values, [5.0, 99.5])
    scale = max(float(p_high - p_low) / 10.0, 1e-6)

    # Compress the enormous XRFD dynamic range so the direct beam / strongest
    # reflection does not dominate the symmetry score.
    work = np.maximum(small - np.float32(p_low), 0.0)
    work = np.arcsinh(work / np.float32(scale)).astype(np.float32, copy=False)

    # Suppress the dimmest background-dominated pixels. This is used only for
    # geometry detection and never changes the quantitative image.
    floor = np.percentile(work[valid], float(settings.intensity_floor_percentile))
    work = np.maximum(work - np.float32(floor), 0.0)
    work[~valid] = 0.0

    mean = np.mean(work[valid], dtype=np.float64)
    work = work - np.float32(mean)
    std = np.std(work[valid], dtype=np.float64)
    if std > 0:
        work = work / np.float32(std)
    work[~valid] = 0.0
    return work.astype(np.float32, copy=False), valid, step


def _parabolic_peak_offset(corr, peak, axis):
    i0 = peak[axis]
    n = corr.shape[axis]
    im = list(peak); ip = list(peak)
    im[axis] = (i0 - 1) % n
    ip[axis] = (i0 + 1) % n
    fm = float(corr[tuple(im)])
    f0 = float(corr[peak])
    fp = float(corr[tuple(ip)])
    denom = fm - 2.0 * f0 + fp
    if abs(denom) < 1e-12:
        return 0.0
    delta = 0.5 * (fm - fp) / denom
    return float(np.clip(delta, -0.5, 0.5))


def _center_from_centrosymmetry(work, initial_center_ds, search_radius_ds):
    moving = np.flipud(np.fliplr(work))
    corr = irfft2(rfft2(work) * np.conj(rfft2(moving)), s=work.shape)
    h, w = work.shape
    array_center = np.array([(h - 1) / 2.0, (w - 1) / 2.0], dtype=float)

    expected_shift = 2.0 * (np.asarray(initial_center_ds, dtype=float) - array_center)
    radius = max(1, int(np.ceil(2.0 * search_radius_ds)))
    ey, ex = np.round(expected_shift).astype(int)

    best_peak = None
    best_value = -np.inf
    for sy in range(ey - radius, ey + radius + 1):
        iy = sy % h
        for sx in range(ex - radius, ex + radius + 1):
            ix = sx % w
            value = float(corr[iy, ix])
            if value > best_value:
                best_value = value
                best_peak = (iy, ix)

    shift = np.array(best_peak, dtype=float)
    for dim, n in enumerate(work.shape):
        if shift[dim] > n // 2:
            shift[dim] -= n

    shift += np.array([
        _parabolic_peak_offset(corr, best_peak, 0),
        _parabolic_peak_offset(corr, best_peak, 1),
    ])

    center = array_center + 0.5 * shift
    norm = np.sqrt(
        np.sum(work * work, dtype=np.float64)
        * np.sum(moving * moving, dtype=np.float64)
    )
    score = float(best_value / norm) if norm > 0 else np.nan
    return center, score


def _pearson(a, b):
    if a.size < 100:
        return np.nan
    aa = a - np.mean(a, dtype=np.float64)
    bb = b - np.mean(b, dtype=np.float64)
    denom = np.sqrt(
        np.sum(aa * aa, dtype=np.float64)
        * np.sum(bb * bb, dtype=np.float64)
    )
    if denom <= 0:
        return np.nan
    return float(np.sum(aa * bb, dtype=np.float64) / denom)


def _inversion_score(work, valid, center, sample_target=80000):
    h, w = work.shape
    step = max(1, int(np.sqrt(work.size / max(4096, int(sample_target)))))
    yy, xx = np.indices(work.shape, dtype=np.float32)
    yy = yy[::step, ::step].ravel()
    xx = xx[::step, ::step].ravel()
    a = work[::step, ::step].ravel()
    source_valid = valid[::step, ::step].ravel()

    ys = 2.0 * float(center[0]) - yy
    xs = 2.0 * float(center[1]) - xx
    inside = (
        (ys >= 0) & (ys <= h - 1)
        & (xs >= 0) & (xs <= w - 1)
        & source_valid
    )

    b = map_coordinates(work, [ys, xs], order=1, mode='constant', cval=np.nan)
    valid_mirror = map_coordinates(
        valid.astype(np.float32), [ys, xs], order=0,
        mode='constant', cval=0.0, prefilter=False,
    ) > 0.5
    good = inside & valid_mirror & np.isfinite(b)
    return _pearson(a[good], b[good])


def _refine_center(work, valid, center0, settings, downsample_step):
    # Convert user-facing full-resolution settings to the reduced image.
    radius = max(0.25, float(settings.center_refine_radius_px) / downsample_step)
    increment = max(0.125, float(settings.center_refine_step_px) / downsample_step)
    offsets = np.arange(-radius, radius + 0.5 * increment, increment)

    best = np.asarray(center0, dtype=float).copy()
    best_score = -np.inf
    for dy in offsets:
        for dx in offsets:
            candidate = np.asarray(center0, dtype=float) + (dy, dx)
            score = _inversion_score(work, valid, candidate)
            if np.isfinite(score) and score > best_score:
                best_score = score
                best = candidate
    return best, float(best_score)


def _reflection_score(
    work, valid, center, angle_deg,
    central_exclusion_ds=0.0,
    sample_target=50000,
):
    h, w = work.shape
    stride = max(1, int(np.sqrt(work.size / max(4096, int(sample_target)))))

    yy, xx = np.indices(work.shape, dtype=np.float32)
    yy = yy[::stride, ::stride].ravel()
    xx = xx[::stride, ::stride].ravel()
    a_all = work[::stride, ::stride].ravel()
    valid_all = valid[::stride, ::stride].ravel()

    ry = yy - np.float32(center[0])
    rx = xx - np.float32(center[1])
    if central_exclusion_ds > 0:
        valid_all &= (rx * rx + ry * ry) >= central_exclusion_ds ** 2

    def score_axis(theta):
        t = np.deg2rad(theta)
        ux = np.float32(np.cos(t))
        uy = np.float32(np.sin(t))
        dot = rx * ux + ry * uy
        rpx = 2.0 * ux * dot - rx
        rpy = 2.0 * uy * dot - ry
        xs = np.float32(center[1]) + rpx
        ys = np.float32(center[0]) + rpy

        inside = (
            (xs >= 0) & (xs <= w - 1)
            & (ys >= 0) & (ys <= h - 1)
            & valid_all
        )
        b = map_coordinates(work, [ys, xs], order=1, mode='constant', cval=np.nan)
        valid_mirror = map_coordinates(
            valid.astype(np.float32), [ys, xs], order=0,
            mode='constant', cval=0.0, prefilter=False,
        ) > 0.5
        good = inside & valid_mirror & np.isfinite(b)
        return _pearson(a_all[good], b[good])

    s1 = score_axis(angle_deg)
    s2 = score_axis(angle_deg + 90.0)
    scores = [s for s in (s1, s2) if np.isfinite(s)]
    return float(np.mean(scores)) if scores else np.nan


def _angle_distance_180(a, b):
    return abs((float(a) - float(b) + 90.0) % 180.0 - 90.0)


def _detect_angle(work, valid, center, initial_angle_deg, settings, step):
    coarse_step = max(0.25, float(settings.coarse_angle_step_deg))
    fine_step = max(0.02, float(settings.fine_angle_step_deg))
    exclusion = max(0.0, float(settings.central_exclusion_radius_px) / step)

    # Because the score includes both perpendicular mirror axes, the objective
    # is 90-degree periodic. Scan only one 90-degree interval, then choose the
    # equivalent axis closest to the user's current fiber-axis estimate.
    coarse_angles = np.arange(0.0, 90.0, coarse_step, dtype=float)
    coarse_scores = np.asarray([
        _reflection_score(
            work, valid, center, a,
            central_exclusion_ds=exclusion,
            sample_target=settings.sample_target,
        )
        for a in coarse_angles
    ], dtype=float)

    if not np.any(np.isfinite(coarse_scores)):
        raise ValueError('Could not obtain a valid mirror-symmetry angle score.')

    coarse_best = float(coarse_angles[np.nanargmax(coarse_scores)])
    fine_angles = np.arange(
        coarse_best - coarse_step,
        coarse_best + coarse_step + 0.5 * fine_step,
        fine_step,
        dtype=float,
    )
    fine_angles = np.mod(fine_angles, 90.0)
    fine_angles = np.unique(np.round(fine_angles, 6))
    fine_scores = np.asarray([
        _reflection_score(
            work, valid, center, a,
            central_exclusion_ds=exclusion,
            sample_target=settings.sample_target,
        )
        for a in fine_angles
    ], dtype=float)

    base = float(fine_angles[np.nanargmax(fine_scores)])
    candidates = [base % 180.0, (base + 90.0) % 180.0]
    chosen = min(candidates, key=lambda a: _angle_distance_180(a, initial_angle_deg))
    return (
        float(chosen),
        float(np.nanmax(fine_scores)),
        coarse_angles,
        coarse_scores,
        fine_angles,
        fine_scores,
    )


def detect_beam_center_and_fiber_axis(
    image,
    mask=None,
    initial_center_x=None,
    initial_center_y=None,
    initial_fiber_angle_deg=90.0,
    settings: AxisDetectionSettings | None = None,
):
    """Detect a symmetry center and the pair of fiber-pattern mirror axes.

    The detector uses centrosymmetry to estimate the beam center, then searches
    mirror-axis orientation on a dynamic-range-compressed copy of the corrected
    image. It does not modify the quantitative image.

    The two perpendicular symmetry axes are mathematically ambiguous; the
    result reported as the fiber axis is the equivalent axis closest to the
    user's current angle estimate.
    """
    if settings is None:
        settings = AxisDetectionSettings()

    image = np.asarray(image, dtype=np.float32)
    h, w = image.shape
    if initial_center_x is None:
        initial_center_x = (w - 1) / 2.0
    if initial_center_y is None:
        initial_center_y = (h - 1) / 2.0

    work, valid, step = _prepare_image(image, mask, settings)
    initial_ds = np.array([
        float(initial_center_y) / step,
        float(initial_center_x) / step,
    ])

    center_ds, fft_score = _center_from_centrosymmetry(
        work,
        initial_center_ds=initial_ds,
        search_radius_ds=float(settings.center_search_radius_px) / step,
    )
    center_ds, center_score = _refine_center(
        work, valid, center_ds, settings, step,
    )

    (
        fiber_angle,
        mirror_score,
        coarse_angles,
        coarse_scores,
        fine_angles,
        fine_scores,
    ) = _detect_angle(
        work,
        valid,
        center_ds,
        initial_fiber_angle_deg,
        settings,
        step,
    )

    # Prefer the direct inversion score after local refinement. If it fails,
    # retain the FFT-normalized score as a diagnostic fallback.
    if not np.isfinite(center_score):
        center_score = fft_score

    return AxisDetectionResult(
        center_x=float(center_ds[1] * step),
        center_y=float(center_ds[0] * step),
        fiber_angle_deg=float(fiber_angle),
        center_symmetry_score=float(center_score),
        mirror_symmetry_score=float(mirror_score),
        coarse_angles_deg=coarse_angles,
        coarse_scores=coarse_scores,
        fine_angles_deg=fine_angles,
        fine_scores=fine_scores,
        downsample_step=int(step),
        settings=settings,
    )
