from __future__ import annotations

from dataclasses import dataclass, asdict
from math import cos, sin, radians
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter1d
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


@dataclass
class LayerLine:
    index: int
    offset_px: float
    measured_offset_px: float | None = None
    intensity: float | None = None
    residual_px: float | None = None

    def to_dict(self):
        return asdict(self)


@dataclass
class LayerLineFitResult:
    center_x: float
    center_y: float
    fiber_angle_deg: float
    count_from_equator_to_anchor: int
    anchor_offset_px: float
    nominal_spacing_px: float
    lines: list[LayerLine]
    axial_positions_px: np.ndarray
    axial_profile: np.ndarray

    def to_dict(self):
        return {
            "center_x": self.center_x,
            "center_y": self.center_y,
            "fiber_angle_deg": self.fiber_angle_deg,
            "count_from_equator_to_anchor": self.count_from_equator_to_anchor,
            "anchor_offset_px": self.anchor_offset_px,
            "nominal_spacing_px": self.nominal_spacing_px,
            "lines": [line.to_dict() for line in self.lines],
        }


def fiber_basis(angle_deg: float):
    theta = radians(float(angle_deg))
    u_parallel = np.array([cos(theta), sin(theta)], dtype=float)
    u_perp = np.array([-sin(theta), cos(theta)], dtype=float)
    return u_parallel, u_perp


def detector_to_fiber(x, y, center_x, center_y, angle_deg):
    u_parallel, u_perp = fiber_basis(angle_deg)
    dx = np.asarray(x, dtype=float) - float(center_x)
    dy = np.asarray(y, dtype=float) - float(center_y)
    parallel = dx * u_parallel[0] + dy * u_parallel[1]
    perp = dx * u_perp[0] + dy * u_perp[1]
    return parallel, perp


def fiber_to_detector(parallel, perp, center_x, center_y, angle_deg):
    u_parallel, u_perp = fiber_basis(angle_deg)
    parallel = np.asarray(parallel, dtype=float)
    perp = np.asarray(perp, dtype=float)
    x = float(center_x) + parallel * u_parallel[0] + perp * u_perp[0]
    y = float(center_y) + parallel * u_parallel[1] + perp * u_perp[1]
    return x, y


def _max_axis_extent(shape, center_x, center_y):
    h, w = shape
    return float(np.hypot(max(center_x, w - 1 - center_x), max(center_y, h - 1 - center_y)))


def sample_fiber_strip(
    image,
    center_x: float,
    center_y: float,
    fiber_angle_deg: float,
    parallel_offset_px: float,
    half_width_px: float = 2.0,
    max_perp_px: float | None = None,
    perp_step_px: float = 1.0,
    parallel_step_px: float = 1.0,
):
    """Sample a strip at constant fiber-parallel coordinate.

    Returns perpendicular coordinates and the mean intensity through the strip.
    The source image is not rotated or modified; interpolation is used only for
    this analysis profile.
    """
    image = np.asarray(image, dtype=np.float32)
    h, w = image.shape
    if max_perp_px is None or max_perp_px <= 0:
        max_perp_px = _max_axis_extent(image.shape, center_x, center_y)

    perp = np.arange(-max_perp_px, max_perp_px + perp_step_px, perp_step_px, dtype=np.float32)
    parallel = np.arange(
        float(parallel_offset_px) - float(half_width_px),
        float(parallel_offset_px) + float(half_width_px) + parallel_step_px,
        parallel_step_px,
        dtype=np.float32,
    )
    pp, qq = np.meshgrid(parallel, perp, indexing="ij")
    x, y = fiber_to_detector(pp, qq, center_x, center_y, fiber_angle_deg)
    sampled = map_coordinates(
        image,
        [y, x],
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )
    valid = np.isfinite(sampled)
    counts = np.sum(valid, axis=0)
    totals = np.nansum(sampled, axis=0, dtype=np.float64)
    profile = np.full(counts.shape, np.nan, dtype=np.float32)
    good = counts > 0
    profile[good] = (totals[good] / counts[good]).astype(np.float32)
    return perp.astype(np.float32), profile


def axial_layer_profile(
    image,
    center_x: float,
    center_y: float,
    fiber_angle_deg: float,
    inner_perp_px: float = 20.0,
    outer_perp_px: float | None = None,
    parallel_step_px: float = 1.0,
    perp_step_px: float = 4.0,
    smooth_sigma_px: float = 1.5,
):
    """Build a 1-D layer-line detection profile along the fiber axis.

    Intensities are averaged across both off-meridional sides while excluding
    a central band around the meridian. This makes the detector less sensitive
    to a single intense meridional spot when detecting layer-line bands.
    """
    image = np.asarray(image, dtype=np.float32)
    max_extent = _max_axis_extent(image.shape, center_x, center_y)
    if outer_perp_px is None or outer_perp_px <= inner_perp_px:
        outer_perp_px = max_extent * 0.75

    parallel = np.arange(-max_extent, max_extent + parallel_step_px, parallel_step_px, dtype=np.float32)
    left = np.arange(-outer_perp_px, -inner_perp_px + perp_step_px, perp_step_px, dtype=np.float32)
    right = np.arange(inner_perp_px, outer_perp_px + perp_step_px, perp_step_px, dtype=np.float32)
    perp = np.concatenate([left, right])
    if perp.size < 2:
        raise ValueError("The selected perpendicular integration range is too small.")

    pp, qq = np.meshgrid(parallel, perp, indexing="ij")
    x, y = fiber_to_detector(pp, qq, center_x, center_y, fiber_angle_deg)
    sampled = map_coordinates(
        image,
        [y, x],
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )
    # Compress extreme dynamic range without changing the source array.
    sampled = np.sign(sampled) * np.sqrt(np.abs(sampled))
    valid = np.isfinite(sampled)
    counts = np.sum(valid, axis=1)
    totals = np.nansum(sampled, axis=1, dtype=np.float64)
    profile = np.full(counts.shape, np.nan, dtype=np.float32)
    good = counts > 0
    profile[good] = (totals[good] / counts[good]).astype(np.float32)
    finite = np.isfinite(profile)
    if np.any(finite):
        fill = float(np.nanmedian(profile[finite]))
        work = np.nan_to_num(profile, nan=fill)
        if smooth_sigma_px > 0:
            work = gaussian_filter1d(work, float(smooth_sigma_px) / max(parallel_step_px, 1e-6))
        profile = work.astype(np.float32)
    return parallel, profile


def fit_evenly_spaced_layer_lines(
    image,
    center_x: float,
    center_y: float,
    fiber_angle_deg: float,
    count_from_equator_to_anchor: int,
    anchor_offset_px: float,
    refine_radius_px: float = 8.0,
    inner_perp_px: float = 20.0,
    outer_perp_px: float | None = None,
    symmetric: bool = True,
):
    """Fit a layer-line ladder with line 0 at the equator and the last line at an anchor.

    ``count_from_equator_to_anchor`` includes the equator. For example, 7 means
    line 0 is the equator and line 6 is the anchor reflection.
    """
    count = int(count_from_equator_to_anchor)
    if count < 2:
        raise ValueError("At least two layer lines (including the equator) are required.")
    anchor = float(anchor_offset_px)
    if abs(anchor) < 1e-6:
        raise ValueError("Anchor offset must be non-zero.")

    sign = 1.0 if anchor >= 0 else -1.0
    anchor_abs = abs(anchor)
    spacing = anchor_abs / float(count - 1)

    pos, profile = axial_layer_profile(
        image,
        center_x,
        center_y,
        fiber_angle_deg,
        inner_perp_px=inner_perp_px,
        outer_perp_px=outer_perp_px,
    )

    lines: list[LayerLine] = []
    indices = list(range(0, count))
    if symmetric:
        indices = list(range(-(count - 1), count))

    for n in indices:
        nominal = float(n) * spacing * sign
        if n == 0:
            measured = 0.0
            intensity = float(np.interp(0.0, pos, profile))
        else:
            lo = nominal - float(refine_radius_px)
            hi = nominal + float(refine_radius_px)
            mask = (pos >= min(lo, hi)) & (pos <= max(lo, hi)) & np.isfinite(profile)
            if np.any(mask):
                local_pos = pos[mask]
                local_prof = profile[mask]
                j = int(np.nanargmax(local_prof))
                measured = float(local_pos[j])
                intensity = float(local_prof[j])
            else:
                measured = nominal
                intensity = None
        lines.append(
            LayerLine(
                index=int(n),
                offset_px=nominal,
                measured_offset_px=measured,
                intensity=intensity,
                residual_px=float(measured - nominal),
            )
        )

    return LayerLineFitResult(
        center_x=float(center_x),
        center_y=float(center_y),
        fiber_angle_deg=float(fiber_angle_deg),
        count_from_equator_to_anchor=count,
        anchor_offset_px=anchor,
        nominal_spacing_px=float(spacing),
        lines=lines,
        axial_positions_px=pos,
        axial_profile=profile,
    )


def suggest_anchor_from_axial_profile(
    image,
    center_x: float,
    center_y: float,
    fiber_angle_deg: float,
    min_offset_px: float = 40.0,
    max_offset_px: float | None = None,
    prominence: float | None = None,
):
    pos, profile = axial_layer_profile(image, center_x, center_y, fiber_angle_deg)
    positive = pos >= float(min_offset_px)
    if max_offset_px is not None and max_offset_px > min_offset_px:
        positive &= pos <= float(max_offset_px)
    x = pos[positive]
    y = profile[positive]
    if x.size < 3:
        raise ValueError("Not enough positive-axis pixels to suggest an anchor.")
    if prominence is None or prominence <= 0:
        spread = float(np.nanpercentile(y, 90) - np.nanpercentile(y, 50))
        prominence = max(spread * 0.15, 1e-6)
    peaks, props = find_peaks(y, prominence=float(prominence))
    if peaks.size == 0:
        return float(x[int(np.nanargmax(y))])
    prominences = props.get("prominences", np.ones(peaks.size))
    # Favor strong peaks farther from the equator because the anchor is often a
    # high-order sharp meridional/layer-line feature.
    score = prominences * (1.0 + 0.25 * (x[peaks] / max(float(np.max(x)), 1.0)))
    return float(x[peaks[int(np.argmax(score))]])


def _gaussian(x, amp, center, sigma):
    sigma = max(abs(float(sigma)), 1e-6)
    return float(amp) * np.exp(-0.5 * ((x - float(center)) / sigma) ** 2)


def _multi_gaussian_with_linear_bg(x, *params):
    n = (len(params) - 2) // 3
    y = params[-2] + params[-1] * x
    for i in range(n):
        amp, center, sigma = params[3 * i:3 * i + 3]
        y = y + _gaussian(x, amp, center, sigma)
    return y


def fit_profile_peaks(
    x,
    y,
    max_peaks: int = 4,
    prominence: float | None = None,
    min_distance_px: float = 8.0,
):
    """Fit up to ``max_peaks`` Gaussian components plus a linear background."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 10:
        raise ValueError("Not enough finite profile points for peak fitting.")

    baseline = float(np.nanpercentile(y, 20))
    work = y - baseline
    if prominence is None or prominence <= 0:
        prominence = max(float(np.nanpercentile(work, 95) - np.nanpercentile(work, 50)) * 0.15, 1e-9)
    step = float(np.nanmedian(np.diff(x))) if x.size > 1 else 1.0
    distance = max(1, int(round(float(min_distance_px) / max(abs(step), 1e-9))))
    peaks, props = find_peaks(work, prominence=float(prominence), distance=distance)
    if peaks.size == 0:
        raise ValueError("No peaks were detected in this layer-line profile.")
    order = np.argsort(props["prominences"])[::-1][: max(1, int(max_peaks))]
    peaks = peaks[order]
    peaks = peaks[np.argsort(x[peaks])]

    p0 = []
    lower = []
    upper = []
    span = max(float(x[-1] - x[0]), 1.0)
    for p in peaks:
        amp = max(float(work[p]), 1e-6)
        cen = float(x[p])
        sigma = max(2.0 * abs(step), span / 300.0)
        p0.extend([amp, cen, sigma])
        lower.extend([0.0, float(x[0]), max(abs(step) * 0.25, 1e-3)])
        upper.extend([np.inf, float(x[-1]), span / 2.0])
    p0.extend([baseline, 0.0])
    lower.extend([-np.inf, -np.inf])
    upper.extend([np.inf, np.inf])

    popt, pcov = curve_fit(
        _multi_gaussian_with_linear_bg,
        x,
        y,
        p0=p0,
        bounds=(lower, upper),
        maxfev=20000,
    )
    fit = _multi_gaussian_with_linear_bg(x, *popt)
    residual = y - fit
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    out_peaks = []
    perr = np.sqrt(np.diag(pcov)) if pcov.size else np.full_like(popt, np.nan)
    n = (len(popt) - 2) // 3
    for i in range(n):
        amp, center, sigma = popt[3 * i:3 * i + 3]
        ea, ec, es = perr[3 * i:3 * i + 3]
        out_peaks.append({
            "amplitude": float(amp),
            "center_px": float(center),
            "sigma_px": float(abs(sigma)),
            "fwhm_px": float(2.354820045 * abs(sigma)),
            "integrated_intensity": float(amp * abs(sigma) * np.sqrt(2 * np.pi)),
            "amplitude_stderr": float(ea),
            "center_stderr_px": float(ec),
            "sigma_stderr_px": float(es),
        })
    return {
        "peaks": out_peaks,
        "background_intercept": float(popt[-2]),
        "background_slope": float(popt[-1]),
        "r_squared": float(r2),
        "x": x,
        "y": y,
        "fit": fit,
        "residual": residual,
    }
