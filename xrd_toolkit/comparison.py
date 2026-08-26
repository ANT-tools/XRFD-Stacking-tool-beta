from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class ComparisonResult:
    mode: str
    scale_mode: str
    scale_b_to_a: float
    alpha: float
    image: np.ndarray
    correlation: float | None
    rms_difference: float | None
    normalized_rms_difference: float | None
    finite_pixel_count: int

    def to_dict(self):
        d = asdict(self)
        d.pop("image", None)
        return d


def fitted_scale(reference, candidate, mask=None):
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if not np.any(valid):
        return 1.0
    av = a[valid]
    bv = b[valid]
    denom = float(np.dot(bv, bv))
    if denom <= 0:
        return 1.0
    return float(np.dot(av, bv) / denom)


def _scale_from_mode(a, b, scale_mode, manual_scale=1.0, exposure_a=None, exposure_b=None):
    mode = scale_mode.lower().strip()
    if mode == "none":
        return 1.0
    if mode == "manual":
        return float(manual_scale)
    if mode == "least-squares fit":
        return fitted_scale(a, b)
    if mode == "total intensity":
        sa = float(np.nansum(a, dtype=np.float64))
        sb = float(np.nansum(b, dtype=np.float64))
        return sa / sb if sb != 0 else 1.0
    if mode == "exposure":
        if exposure_a is None or exposure_b is None or exposure_a <= 0 or exposure_b <= 0:
            raise ValueError("Positive exposure times for both images are required for exposure scaling.")
        # scale B counts to the exposure basis of A
        return float(exposure_a) / float(exposure_b)
    raise ValueError(f"Unknown comparison scale mode {scale_mode!r}")


def compare_images(
    image_a,
    image_b,
    mode="difference",
    scale_mode="least-squares fit",
    manual_scale=1.0,
    alpha=0.5,
    exposure_a=None,
    exposure_b=None,
):
    a = np.asarray(image_a, dtype=np.float32)
    b = np.asarray(image_b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"Comparison requires matching shapes; got {a.shape} and {b.shape}.")

    scale = _scale_from_mode(a, b, scale_mode, manual_scale, exposure_a, exposure_b)
    bs = b.astype(np.float64) * scale
    af = a.astype(np.float64)
    valid = np.isfinite(af) & np.isfinite(bs)
    if not np.any(valid):
        raise ValueError("The two images have no common finite pixels.")

    mode_key = mode.lower().strip()
    if mode_key == "difference":
        out = af - bs
    elif mode_key == "absolute difference":
        out = np.abs(af - bs)
    elif mode_key == "ratio":
        out = np.full(a.shape, np.nan, dtype=np.float64)
        good = valid & (np.abs(bs) > np.finfo(np.float32).eps)
        out[good] = af[good] / bs[good]
    elif mode_key == "overlay":
        alpha = float(np.clip(alpha, 0.0, 1.0))
        out = (1.0 - alpha) * af + alpha * bs
    else:
        raise ValueError(f"Unknown comparison mode {mode!r}")

    av = af[valid]
    bv = bs[valid]
    corr = None
    if av.size > 1 and np.std(av) > 0 and np.std(bv) > 0:
        corr = float(np.corrcoef(av, bv)[0, 1])
    diff = av - bv
    rms = float(np.sqrt(np.mean(diff * diff)))
    scale_ref = float(np.sqrt(np.mean(av * av)))
    nrms = rms / scale_ref if scale_ref > 0 else None

    return ComparisonResult(
        mode=mode,
        scale_mode=scale_mode,
        scale_b_to_a=float(scale),
        alpha=float(alpha),
        image=np.asarray(out, dtype=np.float32),
        correlation=corr,
        rms_difference=rms,
        normalized_rms_difference=nrms,
        finite_pixel_count=int(np.count_nonzero(valid)),
    )
