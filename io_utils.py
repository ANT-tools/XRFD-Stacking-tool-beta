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
