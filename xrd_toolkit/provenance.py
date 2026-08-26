from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json


SCHEMA_NAME = "xrd_image_toolkit_provenance"
SCHEMA_VERSION = "1.0"
TEXT_MAGIC = "XRD-IMAGE-TOOLKIT-PROVENANCE-TEXT 1.0"


FIELD_DESCRIPTIONS = {
    # provenance
    "provenance.schema_name":
        "Stable identifier for the provenance data model used by this toolkit.",
    "provenance.schema_version":
        "Version of the provenance schema. Use this when writing scripts that read these records.",
    "provenance.generated_utc":
        "UTC timestamp at which this provenance record was generated.",
    "provenance.toolkit":
        "Toolkit name/version that created the exported data product and provenance record.",

    # product
    "product.product_type":
        "Scientific role of the exported array. A detector-coordinate corrected image is intended to retain the native detector geometry.",
    "product.output_file":
        "Path of the exported data file described by this provenance record, when available.",
    "product.output_file_sha256":
        "SHA-256 checksum of the exported file. Recalculate it later to verify that the file has not changed.",
    "product.array_rows":
        "Number of rows in the exported numerical array. NumPy axis 0 corresponds to detector row for detector-native products.",
    "product.array_columns":
        "Number of columns in the exported numerical array. NumPy axis 1 corresponds to detector column for detector-native products.",
    "product.array_dtype_in_memory":
        "Numerical dtype of the current in-memory array before export.",
    "product.exported_tiff_dtype":
        "Numerical dtype written to the exported TIFF by the toolkit.",
    "product.source_kind":
        "Whether the current working image originated from one TIFF or from a temporal stack of TIFF accumulations.",
    "product.source_file":
        "Original source TIFF path when the current working image came directly from one file.",
    "product.source_label":
        "Human-readable label for the current working image.",

    # coordinates
    "coordinate_system.coordinate_system":
        "Coordinate convention of the exported array. detector_native means array indices are intended to map to the original detector row/column coordinates.",
    "coordinate_system.axis_0":
        "Meaning of NumPy/TIFF array axis 0.",
    "coordinate_system.axis_1":
        "Meaning of NumPy/TIFF array axis 1.",
    "coordinate_system.array_origin":
        "Meaning of array index [0,0]. upper_left means row numbers increase downward and column numbers increase to the right in the stored array.",
    "coordinate_system.rotation_applied":
        "True if a spatial rotation was applied before export. A native PONI-calibrated detector array normally requires this to be false.",
    "coordinate_system.flip_applied":
        "True if a horizontal or vertical flip was applied before export. A native PONI-calibrated detector array normally requires this to be false.",
    "coordinate_system.crop_applied":
        "True if detector rows/columns were removed before export. Cropping changes the pixel coordinate origin unless calibration is updated accordingly.",
    "coordinate_system.interpolation_applied":
        "True if the detector image was spatially resampled/interpolated. Native detector-coordinate products should normally be false.",
    "coordinate_system.registration_applied":
        "True if temporal frames were shifted during registration before stacking. Even integer shifts change the correspondence between output index and original detector pixel.",
    "coordinate_system.symmetry_transform_applied":
        "True for quadrant-folded/symmetrized products. Such products are no longer in the original detector coordinate frame.",
    "coordinate_system.detector_native_orientation":
        "Summary flag: true only when no rotation, flip, crop, interpolation, registration, or symmetry remapping has been applied.",

    # intensity
    "intensity_processing.dark_subtraction_applied":
        "Whether a dark/reference detector offset was subtracted. This changes intensity values but not detector coordinates.",
    "intensity_processing.flat_field_applied":
        "Whether flat-field sensitivity correction was applied. This changes quantitative intensity values but not detector coordinates.",
    "intensity_processing.background_subtraction_applied":
        "Whether a measured blank/background image was subtracted.",
    "intensity_processing.background_scale":
        "Scale factor used for measured blank/background subtraction.",
    "intensity_processing.monitor_normalization_applied":
        "Whether intensities were divided by the configured exposure/monitor normalization value.",
    "intensity_processing.monitor_value":
        "Exposure/monitor normalization divisor used when monitor normalization is enabled.",
    "intensity_processing.hot_pixel_mask_applied":
        "Whether automatically detected hot pixels were masked.",
    "intensity_processing.saturation_mask_applied":
        "Whether saturated detector pixels were masked.",
    "intensity_processing.beamstop_mask_applied":
        "Whether the configured direct-beam/beamstop region was masked.",
    "intensity_processing.masked_pixel_count":
        "Number of pixels marked invalid in the quantitative corrected array.",
    "intensity_processing.enhancement_filter_applied":
        "Whether display/feature-enhancement filtering is present in this exported product. It should be false for a corrected detector-coordinate TIFF.",
    "intensity_processing.display_transform_applied":
        "Whether log/sqrt/asinh/gamma/histogram/tone-curve display processing was written into this numerical product. It should be false for corrected detector data.",
    "intensity_processing.display_intensity_rescaling_applied":
        "Whether display normalization/contrast rescaling was baked into the exported numerical product. It should be false for quantitative corrected TIFF data.",
    "intensity_processing.custom_tone_curve_applied":
        "Whether the user-defined tone curve was baked into the exported numerical product. It should be false for quantitative corrected TIFF data.",

    # stacking
    "stacking.temporal_stack_used":
        "Whether multiple experimental TIFF accumulations were combined to create the working detector image.",
    "stacking.stack_method":
        "Pixel-combination algorithm used across repeated accumulations.",
    "stacking.frame_count":
        "Number of included experimental accumulations in the temporal stack.",
    "stacking.registration_enabled":
        "Whether frame-to-frame translational registration was enabled during stacking.",
    "stacking.input_files":
        "List of TIFF files included in the temporal stack.",
    "stacking.approximate_ideal_snr_gain":
        "Idealized random-noise SNR gain reported for methods where sqrt(N) scaling is meaningful; systematic errors are not removed by this factor.",
    "stacking.input_correction_state":
        "Declared correction state of the input frames before stacking, used to document whether upstream dark/flat correction was already performed.",
    "stacking.frame_normalization_mode":
        "Per-frame normalization applied before stacking. exposure means frames were rescaled to a common reference exposure.",
    "stacking.reference_exposure_seconds":
        "Reference integration time used when exposure normalization is enabled; output intensities are counts equivalent to this exposure.",
    "stacking.frame_exposures_seconds":
        "Per-file exposure times selected for stacking. These may come from filenames, manual assignment, CSV import, or a confirmed detector header.",
    "stacking.frame_exposure_sources":
        "Per-file provenance of the selected exposure value, such as filename, manual, CSV, or confirmed detector header.",
    "stacking.frame_exposure_candidates":
        "Per-file diagnostic exposure candidates retained alongside the selected value, including filename exposure, header exposure, MarCCD integration time, and mismatch status.",

    # references
    "reference_frames.dark_file":
        "Dark/reference frame loaded in the correction pipeline.",
    "reference_frames.flat_file":
        "Flat-field frame loaded in the correction pipeline.",
    "reference_frames.background_file":
        "Blank/background frame loaded in the correction pipeline.",

    # fiber geometry
    "fiber_geometry.beam_center_x_px":
        "Current beam-center column coordinate in detector pixels.",
    "fiber_geometry.beam_center_y_px":
        "Current beam-center row coordinate in detector pixels.",
    "fiber_geometry.fiber_angle_deg_from_positive_x":
        "Current fiber-axis angle measured in image coordinates from +x (detector-column direction).",
    "fiber_geometry.strip_width_px":
        "Width of the strip used by the toolkit for meridional/equatorial line-profile averaging.",

    # calibration
    "calibration.intended_geometry_calibration":
        "Human-readable name of the geometry/PONI calibration this detector-coordinate product is intended to be used with.",
    "calibration.compatibility_status":
        "Toolkit assessment of whether the array geometry is suitable for the named calibration. intended_unverified means the label is recorded but the PONI file itself has not been audited.",
    "calibration.expected_detector_rows":
        "Expected detector row count associated with the intended calibration.",
    "calibration.expected_detector_columns":
        "Expected detector column count associated with the intended calibration.",
    "calibration.shape_matches_expected_detector":
        "True when the exported array dimensions match the expected detector dimensions recorded above.",
    "calibration.safe_detector_geometry_for_existing_poni":
        "True only when the array shape matches expectations and no spatial operation has broken native detector row/column coordinates.",
    "calibration.compatibility_note":
        "Human-readable statement describing what must be true for the named calibration to remain valid.",

    # header
    "tiff_header.original_tiff_header_preserved":
        "Whether original acquisition TIFF tags/header metadata were copied verbatim into the new TIFF. The current toolkit writes a new TIFF and does not preserve the original acquisition header.",
    "tiff_header.export_header_generated_by_tifffile":
        "True when the exported TIFF container/header was freshly generated by the Python tifffile library.",
    "tiff_header.analysis_implication":
        "Reminder that acquisition metadata needed later should be taken from the original files and provenance records rather than assumed to exist in the exported TIFF header.",

    # guidance
    "analysis_guidance.recommended_for_poni_geometry":
        "Whether the spatial array geometry is suitable for direct use with a PONI calibrated against the same native detector orientation.",
    "analysis_guidance.recommended_for_quantitative_integration":
        "Whether this product is intended to retain quantitative corrected intensities rather than display-only transformed values.",
    "analysis_guidance.use_symmetrized_product_with_original_poni":
        "Should remain false: quadrant-folded products are in a remapped equator/meridian coordinate frame and must not be treated as native detector pixels.",
    "analysis_guidance.use_png_for_quantitative_analysis":
        "Should remain false: PNG exports are rendered display products rather than full-dynamic-range detector arrays.",
    "analysis_guidance.note":
        "Plain-language interpretation of how this file should be used in downstream diffraction analysis.",
}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    path = Path(path)
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)

    return digest.hexdigest()


def write_provenance_json(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _json_value(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def write_provenance_text(path: str | Path, payload: dict) -> Path:
    """
    Write an annotated but machine-readable provenance text file.

    File contract
    -------------
    * UTF-8 text.
    * Lines starting with # are comments.
    * [section] starts a section.
    * Data lines are: key = JSON_VALUE
    * JSON_VALUE follows standard JSON syntax, so booleans are true/false,
      null is null, strings are quoted, and lists/dicts remain structured.

    ``read_provenance_text`` provides a reference parser.
    """
    path = Path(path)

    lines = [
        f"# {TEXT_MAGIC}",
        "#",
        "# HUMAN + MACHINE READABLE FORMAT",
        "# Comments beginning with '#' explain scientific meaning and may be ignored by parsers.",
        "# Each data line is `key = JSON_VALUE`; values use standard JSON syntax.",
        "# Section names in square brackets become top-level JSON objects.",
        "# A reference parser is included as xrd_toolkit.provenance.read_provenance_text().",
        "#",
    ]

    for section, values in payload.items():
        if not isinstance(values, dict):
            values = {"value": values}

        lines.append(f"[{section}]")

        for key, value in values.items():
            description = FIELD_DESCRIPTIONS.get(f"{section}.{key}")

            if description:
                # Wrap comments for readability while keeping parser-simple syntax.
                words = description.split()
                current = "# "
                for word in words:
                    if len(current) + len(word) + 1 > 100:
                        lines.append(current.rstrip())
                        current = "# " + word + " "
                    else:
                        current += word + " "
                if current.strip() != "#":
                    lines.append(current.rstrip())

            lines.append(f"{key} = {_json_value(value)}")
            lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def read_provenance_text(path: str | Path) -> dict:
    """
    Reference parser for the toolkit's annotated provenance .txt format.
    """
    path = Path(path)
    payload: dict[str, dict] = {}
    current_section: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1].strip()
            if not current_section:
                raise ValueError("Empty section name in provenance text.")
            payload.setdefault(current_section, {})
            continue

        if current_section is None:
            raise ValueError(
                f"Data line encountered before a section header: {raw_line!r}"
            )

        if "=" not in line:
            raise ValueError(f"Invalid provenance data line: {raw_line!r}")

        key, value_text = line.split("=", 1)
        key = key.strip()
        value_text = value_text.strip()

        if not key:
            raise ValueError(f"Missing key in provenance line: {raw_line!r}")

        payload[current_section][key] = json.loads(value_text)

    return payload


def write_provenance_bundle(
    base_path: str | Path,
    payload: dict,
) -> tuple[Path, Path]:
    """
    Write matching .json and annotated .txt provenance sidecars.

    If ``base_path`` ends in .json or .txt, the suffix is replaced.
    Otherwise it is treated as the complete shared stem.
    """
    base = Path(base_path)

    if base.suffix.lower() in {".json", ".txt"}:
        stem = base.with_suffix("")
    else:
        stem = base

    json_path = stem.with_suffix(".json")
    txt_path = stem.with_suffix(".txt")

    write_provenance_json(json_path, payload)
    write_provenance_text(txt_path, payload)

    return json_path, txt_path
