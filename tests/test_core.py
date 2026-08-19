from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from xrd_toolkit.processing import ProcessingSettings, process_image
from xrd_toolkit.provenance import read_provenance_text, write_provenance_bundle
from xrd_toolkit.symmetry import SymmetrySettings, build_symmetry_average


def test_display_pipeline_preserves_shape():
    image = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    result = process_image(image, ProcessingSettings(display_mode="log"))

    assert result["raw"].shape == image.shape
    assert result["corrected"].shape == image.shape
    assert result["display"].shape == image.shape


def test_four_quadrant_symmetry_is_symmetric():
    n = 65
    c = (n - 1) / 2
    y, x = np.indices((n, n), dtype=np.float32)
    dx = x - c
    dy = y - c
    image = (
        np.exp(-((np.abs(dx) - 14) ** 2 + (np.abs(dy) - 8) ** 2) / 20)
        + 0.2 * np.exp(-(dx**2 + dy**2) / 40)
    ).astype(np.float32)

    result = build_symmetry_average(
        image,
        center_x=c,
        center_y=c,
        fiber_angle_deg=90.0,
        settings=SymmetrySettings(mode="four-quadrant", statistic="mean"),
    )

    assert np.allclose(result.symmetrized, np.fliplr(result.symmetrized), equal_nan=True)
    assert np.allclose(result.symmetrized, np.flipud(result.symmetrized), equal_nan=True)


def test_annotated_provenance_round_trip(tmp_path: Path):
    payload = {
        "product": {"array_rows": 3072, "array_columns": 3072},
        "coordinate_system": {
            "axis_0": "detector_row",
            "axis_1": "detector_column",
            "rotation_applied": False,
        },
    }

    json_path, txt_path = write_provenance_bundle(
        tmp_path / "example.provenance",
        payload,
    )

    assert json.loads(json_path.read_text(encoding="utf-8")) == payload
    assert read_provenance_text(txt_path) == payload
