from pathlib import Path
import tempfile
import numpy as np
import tifffile

from xrd_toolkit.stacking import StackSettings, build_stack
from xrd_toolkit.layer_lines import fit_evenly_spaced_layer_lines, fit_profile_peaks
from xrd_toolkit.roi import ROIShape, measure_roi
from xrd_toolkit.comparison import compare_images
from xrd_toolkit.session import save_session, load_session
from xrd_toolkit.io_utils import parse_exposure_from_filename, select_exposure_from_info


def test_exposure_normalized_stack():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        exposures = [1.0, 2.0, 5.0]
        paths = []
        for i, exposure in enumerate(exposures):
            path = td / f"frame_{i}.tif"
            tifffile.imwrite(path, np.full((32, 32), 100.0 * exposure, dtype=np.float32))
            paths.append(path)
        settings = StackSettings(
            method="exposure-weighted mean",
            normalization_mode="exposure",
            reference_exposure_seconds=1.0,
            frame_exposures={str(p): t for p, t in zip(paths, exposures)},
        )
        result = build_stack(paths, settings)
        assert np.allclose(result.image, 100.0)


def test_layer_line_ladder():
    n = 192
    cx = cy = (n - 1) / 2
    yy, xx = np.indices((n, n), dtype=float)
    image = np.ones((n, n), dtype=np.float32)
    spacing = 18.0
    for k in range(-5, 6):
        image += (30 * np.exp(-0.5 * ((yy - (cy + k * spacing)) / 1.5) ** 2)).astype(np.float32)
    result = fit_evenly_spaced_layer_lines(
        image, cx, cy, 90.0, 6, 5 * spacing,
        refine_radius_px=4.0, inner_perp_px=10, outer_perp_px=70,
    )
    assert abs(result.nominal_spacing_px - spacing) < 1e-6


def test_multi_peak_fit():
    x = np.linspace(-60, 60, 241)
    y = 3 + 45*np.exp(-0.5*((x+18)/4)**2) + 30*np.exp(-0.5*((x-24)/5)**2)
    result = fit_profile_peaks(x, y, max_peaks=2, prominence=5)
    centers = sorted(p["center_px"] for p in result["peaks"])
    assert np.allclose(centers, [-18, 24], atol=0.5)


def test_roi_and_comparison_and_session():
    a = np.arange(10000, dtype=np.float32).reshape(100, 100) + 1
    b = a * 4
    roi = ROIShape("ROI-001", "circle", 50, 50, 60, 50)
    assert measure_roi(a, roi)["pixel_count"] > 0
    result = compare_images(a, b, mode="difference", scale_mode="least-squares fit")
    assert np.nanmax(np.abs(result.image)) < 1e-3
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "session.json"
        save_session(path, {"rois": [roi.to_dict()]})
        assert load_session(path)["rois"][0]["roi_id"] == "ROI-001"


def test_filename_exposure_parser_is_conservative():
    assert parse_exposure_from_filename("S1_3_5s_S030_0_10.tif")["seconds"] == 5.0
    assert parse_exposure_from_filename("S1_3_10s_S031_0_00.tif")["seconds"] == 10.0
    assert parse_exposure_from_filename("S1_3_500ms_S032_0_00.tif")["seconds"] == 0.5
    assert parse_exposure_from_filename("S1_3_0.5s_S033_0_00.tif")["seconds"] == 0.5
    assert parse_exposure_from_filename("S1_3_S029_0_01.tif")["seconds"] is None


def test_exposure_policy_does_not_silently_use_header_by_default():
    info = {
        "filename_exposure_seconds": None,
        "header_exposure_seconds": 1.0,
        "header_exposure_source": "MarCCD header",
    }
    assert select_exposure_from_info(info, "filename only") == (None, None)
    assert select_exposure_from_info(info, "filename → header") == (1.0, "MarCCD header")


def test_frame_exposure_source_is_retained():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        p = td / "sample_5s_001.tif"
        tifffile.imwrite(p, np.full((16, 16), 500.0, dtype=np.float32))
        settings = StackSettings(
            method="mean",
            normalization_mode="exposure",
            reference_exposure_seconds=1.0,
            frame_exposures={str(p): 5.0},
            frame_exposure_sources={str(p): "filename"},
            frame_exposure_candidates={str(p): {
                "filename_exposure_seconds": 5.0,
                "header_exposure_seconds": 5.0,
                "marccd_integration_seconds": 7.7,
                "exposure_mismatch": False,
            }},
        )
        result = build_stack([p], settings)
        stat = result.frame_stats[0]
        assert stat.exposure_seconds == 5.0
        assert stat.exposure_source == "filename"
        assert stat.filename_exposure_seconds == 5.0
        assert stat.header_exposure_seconds == 5.0
