from __future__ import annotations

from pathlib import Path
import json
import os
import numpy as np
import tifffile

TIFF_EXTENSIONS = {".tif", ".tiff"}

def inspect_tiff_output_shape(path: str | Path) -> tuple[int, int]:
    """Return the 2-D image shape produced by ``load_tiff`` without decoding pixels.

    Scientific multi-page TIFFs are treated the same way as ``load_tiff``:
    a 2-D TIFF stays 2-D, while a 3-D frame stack is reduced along axis 0 and
    therefore has output shape ``shape[1:]``.
    """
    path = Path(path)
    with tifffile.TiffFile(path) as tif:
        if not tif.series:
            raise ValueError(f"{path.name}: TIFF contains no readable image series.")
        shape = tuple(int(v) for v in tif.series[0].shape)

    if len(shape) == 2:
        return shape
    if len(shape) == 3:
        output = shape[1:]
        if len(output) != 2:
            raise ValueError(f"{path.name}: unsupported TIFF shape {shape}.")
        return tuple(int(v) for v in output)
    raise ValueError(
        f"{path.name}: expected a 2-D TIFF or a 3-D TIFF frame stack; got shape {shape}."
    )



def default_tiff_workers() -> int:
    # TIFF decoding benefits from several workers, but very high thread counts
    # can compete with NumPy/SciPy threads and disk I/O.
    return max(1, min(8, os.cpu_count() or 1))


def load_tiff(
    path: str | Path,
    stack_mode: str = "mean",
    frame_index: int = 0,
    dtype=np.float32,
    maxworkers: int | None = None,
):
    path = Path(path)
    if maxworkers is None:
        maxworkers = default_tiff_workers()
    arr = tifffile.imread(path, maxworkers=maxworkers)

    metadata = {
        "path": str(path),
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "stack_mode": stack_mode,
        "tiff_decode_workers": int(maxworkers),
    }

    if arr.ndim == 2:
        image = arr
    elif arr.ndim == 3:
        if stack_mode == "mean":
            image = np.mean(arr, axis=0, dtype=np.float64)
        elif stack_mode == "sum":
            image = np.sum(arr, axis=0, dtype=np.float64)
        elif stack_mode == "first":
            image = arr[0]
        elif stack_mode == "index":
            image = arr[frame_index]
            metadata["frame_index"] = frame_index
        else:
            raise ValueError(f"Unknown TIFF stack mode: {stack_mode}")
    else:
        raise ValueError(f"Expected 2-D TIFF or TIFF stack; got shape {arr.shape}")

    return np.asarray(image, dtype=dtype), metadata


def list_tiffs(folder: str | Path):
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in TIFF_EXTENSIONS)


def save_float_tiff(path: str | Path, image):
    tifffile.imwrite(Path(path), np.asarray(image, dtype=np.float32))


def save_mask_tiff(path: str | Path, mask):
    tifffile.imwrite(Path(path), np.asarray(mask, dtype=np.uint8))


def save_csv(path: str | Path, columns, header: str):
    np.savetxt(Path(path), np.column_stack(columns), delimiter=",", header=header, comments="")


def save_json(path: str | Path, payload: dict):
    Path(path).write_text(json.dumps(payload, indent=2))


def parse_exposure_from_filename(path: str | Path) -> dict:
    """Parse an explicit exposure token from a filename.

    Only numbers immediately carrying an exposure unit are accepted, for example
    ``_5s_``, ``_10sec_``, ``_0.5s_``, ``_500ms_``. Bare numbers such as
    ``S029``, ``_0_`` or ``_01`` are intentionally ignored.
    """
    import re

    name = Path(path).name
    pattern = re.compile(
        r"(?i)(?:^|[_\-\s])"
        r"(?P<value>(?:\d+(?:\.\d*)?|\.\d+))"
        r"(?P<unit>milliseconds?|msecs?|ms|seconds?|secs?|sec|s)"
        r"(?=$|[_\-\s.])"
    )
    matches = list(pattern.finditer(name))
    if not matches:
        return {"seconds": None, "token": None, "source": None, "match_count": 0}

    match = matches[0]
    value = float(match.group("value"))
    unit = match.group("unit").lower()
    if unit.startswith("ms") or unit.startswith("millisecond"):
        seconds = value / 1000.0
    else:
        seconds = value
    if not np.isfinite(seconds) or seconds <= 0:
        seconds = None

    return {
        "seconds": float(seconds) if seconds is not None else None,
        "token": match.group(0).strip("_- ."),
        "source": "filename" if seconds is not None else None,
        "match_count": len(matches),
    }


def _standard_tiff_exposure_seconds(path: str | Path) -> float | None:
    """Read a conventional TIFF exposure field when explicitly present.

    Integration-time fields are deliberately not substituted for exposure time.
    """
    path = Path(path)
    candidates = []
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        for key in ("ExposureTime", "Exposure", "exposure_time", "exposure_seconds"):
            tag = page.tags.get(key)
            if tag is not None:
                candidates.append((key, tag.value))

        desc = getattr(page, "description", None)
        if desc:
            text = str(desc).strip()
            try:
                data = json.loads(text)
            except Exception:
                data = None
            if isinstance(data, dict):
                for key in ("ExposureTime", "exposure_time", "exposure_seconds"):
                    if key in data:
                        candidates.append((key, data[key]))
            else:
                import re
                match = re.search(
                    r"(?i)exposure(?:\s*time)?\s*[:=]\s*"
                    r"([0-9.eE+-]+)\s*(ms|msec|milliseconds?|s|sec|seconds?)?",
                    text,
                )
                if match:
                    value = float(match.group(1))
                    unit = (match.group(2) or "s").lower()
                    if unit.startswith("ms") or unit.startswith("millisecond"):
                        value /= 1000.0
                    if np.isfinite(value) and value > 0:
                        return float(value)

    for key, raw in candidates:
        try:
            if isinstance(raw, (tuple, list)) and len(raw) == 2:
                value = float(raw[0]) / float(raw[1])
            else:
                value = float(raw)
            if "ms" in key.lower():
                value /= 1000.0
            if np.isfinite(value) and value > 0:
                return float(value)
        except Exception:
            continue
    return None


def inspect_marccd_times(path: str | Path) -> dict:
    """Best-effort read of Rayonix/MarCCD private-frame timing fields.

    Native MarCCD TIFFs contain a private TIFF tag (34710) pointing to the
    3072-byte frame_header. ``integration_time`` and ``exposure_time`` are
    signed 32-bit millisecond fields at offsets 652 and 656 bytes from that
    frame header. These values are diagnostic and are not automatically trusted
    over an explicit filename token by the GUI.
    """
    import struct

    result = {
        "is_marccd": False,
        "header_offset": None,
        "integration_seconds": None,
        "exposure_seconds": None,
    }
    path = Path(path)
    try:
        with tifffile.TiffFile(path) as tif:
            if not tif.pages:
                return result
            page = tif.pages[0]
            tag = page.tags.get(34710) or page.tags.get("MarCCD")
            if tag is None:
                return result
            frame_offset = int(tag.value)
            endian = "<" if getattr(tif, "byteorder", "<") == "<" else ">"

        with path.open("rb") as handle:
            handle.seek(frame_offset + 652)
            raw = handle.read(8)
        if len(raw) != 8:
            return result

        integration_ms, exposure_ms = struct.unpack(endian + "ii", raw)
        result["is_marccd"] = True
        result["header_offset"] = frame_offset
        max_ms = 7 * 24 * 3600 * 1000
        if 0 < integration_ms <= max_ms:
            result["integration_seconds"] = float(integration_ms) / 1000.0
        if 0 < exposure_ms <= max_ms:
            result["exposure_seconds"] = float(exposure_ms) / 1000.0
    except Exception:
        pass
    return result


def inspect_tiff_exposure_info(path: str | Path) -> dict:
    """Return filename/header exposure candidates and a mismatch diagnostic."""
    path = Path(path)
    filename = parse_exposure_from_filename(path)
    try:
        standard = _standard_tiff_exposure_seconds(path)
    except Exception:
        standard = None
    marccd = inspect_marccd_times(path)

    header_exposure = standard
    header_source = "TIFF metadata" if standard is not None else None
    if header_exposure is None and marccd.get("exposure_seconds") is not None:
        header_exposure = marccd["exposure_seconds"]
        header_source = "MarCCD header"

    filename_exposure = filename.get("seconds")
    mismatch = False
    if filename_exposure is not None and header_exposure is not None:
        tol = max(0.05, 0.02 * max(filename_exposure, header_exposure))
        mismatch = abs(filename_exposure - header_exposure) > tol

    return {
        "path": str(path),
        "filename_exposure_seconds": filename_exposure,
        "filename_exposure_token": filename.get("token"),
        "filename_exposure_match_count": int(filename.get("match_count", 0) or 0),
        "standard_tiff_exposure_seconds": standard,
        "marccd_exposure_seconds": marccd.get("exposure_seconds"),
        "marccd_integration_seconds": marccd.get("integration_seconds"),
        "header_exposure_seconds": header_exposure,
        "header_exposure_source": header_source,
        "exposure_mismatch": bool(mismatch),
    }


def select_exposure_from_info(info: dict, policy: str = "filename only") -> tuple[float | None, str | None]:
    """Select an automatic exposure according to an explicit GUI policy."""
    policy = str(policy or "filename only").strip().lower()
    filename = info.get("filename_exposure_seconds")
    header = info.get("header_exposure_seconds")
    header_source = info.get("header_exposure_source") or "header"

    if policy in {"filename only", "filename"}:
        return (float(filename), "filename") if filename is not None else (None, None)
    if policy in {"filename → header", "filename -> header", "filename then header"}:
        if filename is not None:
            return float(filename), "filename"
        if header is not None:
            return float(header), str(header_source)
        return None, None
    if policy in {"header → filename", "header -> filename", "header then filename"}:
        if header is not None:
            return float(header), str(header_source)
        if filename is not None:
            return float(filename), "filename"
        return None, None
    return (float(filename), "filename") if filename is not None else (None, None)


def inspect_tiff_exposure_seconds(path: str | Path) -> float | None:
    """Compatibility helper: filename exposure first, then header exposure."""
    info = inspect_tiff_exposure_info(path)
    if info.get("filename_exposure_seconds") is not None:
        return float(info["filename_exposure_seconds"])
    if info.get("header_exposure_seconds") is not None:
        return float(info["header_exposure_seconds"])
    return None

