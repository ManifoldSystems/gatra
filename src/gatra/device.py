from __future__ import annotations

from dataclasses import dataclass

import torch


class DeviceError(ValueError):
    pass


@dataclass(frozen=True)
class DeviceInfo:
    device: torch.device
    kind: str
    name: str

    def __str__(self) -> str:
        return f"{self.device} ({self.kind}: {self.name})"


def cuda_available() -> bool:
    return torch.cuda.is_available()


def rocm_available() -> bool:
    return cuda_available() and bool(getattr(torch.version, "hip", None))


def nvidia_available() -> bool:
    return cuda_available() and not bool(getattr(torch.version, "hip", None))


def mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return backend is not None and backend.is_available()


def _cuda_name(index: int | None = None) -> str:
    if not cuda_available():
        return "unavailable"
    if index is None:
        index = torch.cuda.current_device()
    return torch.cuda.get_device_name(index)


def _parse(spec: str) -> tuple[str, int | None]:
    raw = spec.strip().lower()
    if raw in {"auto", "cpu", "mps", "cuda", "rocm", "hip"}:
        return raw, None
    if ":" in raw:
        kind, index_s = raw.split(":", 1)
        if kind in {"cuda", "rocm", "hip"} and index_s.isdigit():
            return kind, int(index_s)
    raise DeviceError(
        f"unknown device '{spec}'. use auto, cpu, mps, cuda, cuda:N, rocm, rocm:N"
    )


def resolve_device(spec: str = "auto") -> DeviceInfo:
    kind, index = _parse(spec)

    if kind == "cpu":
        return DeviceInfo(torch.device("cpu"), "cpu", "cpu")

    if kind == "mps":
        if not mps_available():
            raise DeviceError("MPS is not available on this PyTorch build")
        return DeviceInfo(torch.device("mps"), "mps", "apple-gpu")

    if kind in {"cuda", "rocm", "hip"}:
        if not cuda_available():
            label = "ROCm" if kind in {"rocm", "hip"} else "CUDA"
            raise DeviceError(f"{label} is not available on this PyTorch build")
        if kind in {"rocm", "hip"} and not rocm_available():
            raise DeviceError("ROCm/HIP was requested, but this PyTorch build is not HIP")
        if index is not None and index >= torch.cuda.device_count():
            raise DeviceError(f"GPU index {index} is out of range ({torch.cuda.device_count()} visible)")
        torch_kind = "cuda" if index is None else f"cuda:{index}"
        backend = "rocm" if rocm_available() else "cuda"
        return DeviceInfo(torch.device(torch_kind), backend, _cuda_name(index))

    if nvidia_available() or cuda_available():
        backend = "rocm" if rocm_available() else "cuda"
        return DeviceInfo(torch.device("cuda"), backend, _cuda_name())
    if mps_available():
        return DeviceInfo(torch.device("mps"), "mps", "apple-gpu")
    return DeviceInfo(torch.device("cpu"), "cpu", "cpu")


def pick_device(spec: str = "auto") -> torch.device:
    return resolve_device(spec).device
