#!/usr/bin/env python3
"""Small local benchmark for XRD Image Toolkit CPU/GPU performance."""
from __future__ import annotations

import argparse
import time
import numpy as np

from xrd_toolkit.backend import get_gpu_info
from xrd_toolkit.processing import ProcessingSettings, ProcessingEngine, process_image
from xrd_toolkit.symmetry import SymmetrySettings, build_symmetry_average


def timed(label, fn):
    t0 = time.perf_counter()
    result = fn()
    dt = time.perf_counter() - t0
    print(f"{label:36s} {dt:8.3f} s")
    return result, dt


def make_image(n):
    rng = np.random.default_rng(1234)
    y, x = np.indices((n, n), dtype=np.float32)
    c = (n - 1) / 2
    image = (
        8000*np.exp(-((x-c)**2+(y-c)**2)/(2*12**2))
        + 1800*np.exp(-((np.abs(x-c)-0.18*n)**2)/(2*5**2)-((np.abs(y-c)-0.11*n)**2)/(2*4**2))
        + 100*rng.random((n,n), dtype=np.float32)
    )
    return image.astype(np.float32)


def run_backend(image, backend):
    print(f"\nBackend: {backend.upper()}")
    base = ProcessingSettings(compute_backend=backend, display_mode="log")
    _, display_t = timed("Display-only pipeline", lambda: process_image(image, base))

    filt = ProcessingSettings(
        compute_backend=backend,
        gaussian_background_enabled=True,
        gaussian_background_sigma=40,
        high_pass_enabled=True,
        high_pass_sigma=6,
        display_mode="log",
    )
    result, filter_t = timed("Gaussian background + high-pass", lambda: process_image(image, filt))

    engine = ProcessingEngine()
    engine.process(image, filt)
    filt2 = ProcessingSettings(**filt.to_dict())
    filt2.display_mode = "sqrt"
    _, cache_t = timed("Cached display-only change", lambda: engine.process(image, filt2))

    sym_settings = SymmetrySettings(
        mode="four-quadrant",
        statistic="mean",
        compute_backend=backend,
    )
    _, sym_t = timed(
        "Four-quadrant symmetry average",
        lambda: build_symmetry_average(
            result["corrected"],
            center_x=(image.shape[1]-1)/2,
            center_y=(image.shape[0]-1)/2,
            fiber_angle_deg=90.0,
            settings=sym_settings,
            mask=result["mask"],
        ),
    )
    return {"display":display_t,"filter":filter_t,"cached_display":cache_t,"symmetry":sym_t}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=2048, help="Synthetic square image size")
    args = parser.parse_args()

    info = get_gpu_info()
    print("XRD Image Toolkit benchmark")
    print(f"Synthetic image: {args.size} x {args.size} float32")
    if info.available:
        print(f"GPU: {info.name}")
    else:
        print(f"GPU: unavailable ({info.error})")

    image = make_image(args.size)
    cpu = run_backend(image, "cpu")
    if info.available:
        gpu = run_backend(image, "gpu")
        print("\nGPU / CPU speedup:")
        for key in cpu:
            if gpu[key] > 0:
                print(f"{key:20s} {cpu[key]/gpu[key]:7.2f}x")


if __name__ == "__main__":
    main()
