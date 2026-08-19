# Optional GPU acceleration

XRD Image Toolkit 0.4.0 can use an NVIDIA CUDA GPU through **CuPy** for the
heavier 2-D operations, including Gaussian/median filtering, high-pass and
unsharp operations, symmetry resampling/averaging, and the expensive block
operations used by median or sigma-clipped stacks.

GPU support is optional. The normal `requirements.txt` remains CPU-only so the
toolkit continues to install on machines without CUDA.

## Windows / Linux with NVIDIA CUDA

Install **one** CuPy package matching the CUDA generation available on the
machine:

```bash
# CUDA 12.x
python -m pip install cupy-cuda12x

# CUDA 13.x
python -m pip install cupy-cuda13x
```

If you have a compatible NVIDIA driver but do not want to manage a separate
CUDA Toolkit installation, current CuPy releases also support installing the
CUDA component wheels, for example:

```bash
python -m pip install "cupy-cuda12x[ctk]"
```

Do not install more than one `cupy` / `cupy-cudaXX` package in the same Python
environment.

## In the toolkit

Open **Performance** and choose:

- `auto` — use GPU when CuPy/CUDA is available, otherwise CPU
- `cpu` — always use the optimized NumPy/SciPy path
- `gpu` — require GPU and report an error if it is unavailable

The Performance tab shows the detected GPU and VRAM.

## Benchmark

Run:

```bash
python benchmark_xrd_toolkit.py --size 2048
```

If a CUDA GPU is available the script benchmarks both CPU and GPU and prints
the measured speedups on that machine.
