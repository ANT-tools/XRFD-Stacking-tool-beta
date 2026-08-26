from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class ROIShape:
    roi_id: str
    shape_type: str
    x0: float
    y0: float
    x1: float
    y1: float
    label: str = ""
    visible: bool = True

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


def roi_mask(shape: ROIShape, image_shape):
    h, w = image_shape
    yy, xx = np.ogrid[:h, :w]
    kind = shape.shape_type.lower().strip()

    if kind == "circle":
        cx, cy = float(shape.x0), float(shape.y0)
        r = float(np.hypot(shape.x1 - shape.x0, shape.y1 - shape.y0))
        return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2

    if kind == "ellipse":
        cx = 0.5 * (shape.x0 + shape.x1)
        cy = 0.5 * (shape.y0 + shape.y1)
        rx = max(abs(shape.x1 - shape.x0) * 0.5, 1e-6)
        ry = max(abs(shape.y1 - shape.y0) * 0.5, 1e-6)
        return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0

    if kind == "rectangle":
        xmin, xmax = sorted([float(shape.x0), float(shape.x1)])
        ymin, ymax = sorted([float(shape.y0), float(shape.y1)])
        return (xx >= xmin) & (xx <= xmax) & (yy >= ymin) & (yy <= ymax)

    if kind == "line":
        # A thin 3-pixel analysis band around a line segment.
        x0, y0, x1, y1 = map(float, [shape.x0, shape.y0, shape.x1, shape.y1])
        vx, vy = x1 - x0, y1 - y0
        denom = vx * vx + vy * vy
        if denom <= 0:
            return (xx - x0) ** 2 + (yy - y0) ** 2 <= 2.25
        t = ((xx - x0) * vx + (yy - y0) * vy) / denom
        t = np.clip(t, 0.0, 1.0)
        px = x0 + t * vx
        py = y0 + t * vy
        return (xx - px) ** 2 + (yy - py) ** 2 <= 2.25

    raise ValueError(f"Unsupported ROI shape {shape.shape_type!r}")


def measure_roi(image, shape: ROIShape):
    image = np.asarray(image, dtype=float)
    mask = roi_mask(shape, image.shape)
    values = image[mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "roi_id": shape.roi_id,
            "label": shape.label,
            "shape_type": shape.shape_type,
            "pixel_count": 0,
            "mean": None,
            "median": None,
            "minimum": None,
            "maximum": None,
            "sum": None,
            "std": None,
        }
    return {
        "roi_id": shape.roi_id,
        "label": shape.label,
        "shape_type": shape.shape_type,
        "pixel_count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "sum": float(np.sum(values, dtype=np.float64)),
        "std": float(np.std(values, dtype=np.float64)),
    }
