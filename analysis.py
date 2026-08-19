from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks


def extract_line_profile(
    image,
    center_x,
    center_y,
    angle_deg,
    half_length=None,
    strip_width=5,
    samples=None,
):
    image = np.asarray(image, dtype=float)
    h, w = image.shape

    if half_length is None:
        half_length = 0.5 * np.hypot(h, w)

    if samples is None:
        samples = int(2 * half_length) + 1

    theta = np.deg2rad(angle_deg)
    direction = np.array([np.cos(theta), np.sin(theta)])
    normal = np.array([-np.sin(theta), np.cos(theta)])

    distance = np.linspace(-half_length, half_length, samples)
    offsets = np.linspace(
        -(strip_width - 1) / 2,
        (strip_width - 1) / 2,
        max(1, int(strip_width)),
    )

    stack = []

    for off in offsets:
        xs = center_x + distance * direction[0] + off * normal[0]
        ys = center_y + distance * direction[1] + off * normal[1]

        valid = (
            (xs >= 0) & (xs <= w - 1) &
            (ys >= 0) & (ys <= h - 1)
        )

        coords = np.vstack([ys, xs])

        vals = map_coordinates(
            np.nan_to_num(image, nan=0.0),
            coords,
            order=1,
            mode="constant",
            cval=np.nan,
        )
        vals[~valid] = np.nan
        stack.append(vals)

    stack = np.asarray(stack)
    profile = np.nanmean(stack, axis=0)

    return distance, profile


def extract_meridian_equator(
    image,
    center_x,
    center_y,
    fiber_angle_deg=90.0,
    strip_width=5,
):
    meridian = extract_line_profile(
        image,
        center_x,
        center_y,
        fiber_angle_deg,
        strip_width=strip_width,
    )

    equator = extract_line_profile(
        image,
        center_x,
        center_y,
        fiber_angle_deg - 90.0,
        strip_width=strip_width,
    )

    return {
        "meridian": meridian,
        "equator": equator,
    }


def find_profile_peaks(
    distance,
    intensity,
    prominence=None,
    distance_pixels=None,
):
    y = np.asarray(intensity, dtype=float)
    valid = np.isfinite(y)

    if not np.any(valid):
        return np.array([], dtype=int), {}

    fill = np.nanmedian(y[valid])
    yy = np.nan_to_num(y, nan=fill)

    kwargs = {}

    if prominence is not None:
        kwargs["prominence"] = prominence

    if distance_pixels is not None:
        kwargs["distance"] = distance_pixels

    peaks, props = find_peaks(yy, **kwargs)
    return peaks, props
