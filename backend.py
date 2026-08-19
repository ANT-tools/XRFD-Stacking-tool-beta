from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class GPUInfo:
    available: bool
    name: str | None = None
    total_memory_bytes: int | None = None
    free_memory_bytes: int | None = None
    error: str | None = None

    def to_dict(self):
        return asdict(self)


def get_gpu_info() -> GPUInfo:
    """Return CUDA/CuPy status without making CuPy a required dependency."""
    try:
        import cupy as cp
        count = int(cp.cuda.runtime.getDeviceCount())
        if count < 1:
            return GPUInfo(False, error="CuPy is installed but no CUDA GPU was detected.")
        dev = cp.cuda.Device(0)
        with dev:
            props = cp.cuda.runtime.getDeviceProperties(0)
            raw_name = props.get("name", b"CUDA GPU")
            name = raw_name.decode(errors="replace") if isinstance(raw_name, bytes) else str(raw_name)
            free_b, total_b = cp.cuda.runtime.memGetInfo()
        return GPUInfo(True, name=name, total_memory_bytes=int(total_b), free_memory_bytes=int(free_b))
    except Exception as exc:
        return GPUInfo(False, error=str(exc))


def resolve_backend(requested: str = "auto") -> str:
    requested = (requested or "auto").lower().strip()
    if requested not in {"auto", "cpu", "gpu"}:
        raise ValueError("Backend must be auto, cpu, or gpu.")
    if requested == "cpu":
        return "cpu"
    info = get_gpu_info()
    if requested == "gpu" and not info.available:
        raise RuntimeError(
            "GPU processing was requested but a usable CuPy/CUDA device was not found. "
            f"Details: {info.error or 'unknown error'}"
        )
    return "gpu" if info.available else "cpu"
