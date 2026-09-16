import pytest
import torch

from gatra.device import DeviceError, pick_device, resolve_device


def test_cpu_device() -> None:
    info = resolve_device("cpu")
    assert info.device == torch.device("cpu")
    assert info.kind == "cpu"
    assert pick_device("cpu").type == "cpu"


def test_auto_falls_back_to_available_backend() -> None:
    info = resolve_device("auto")
    assert info.device.type in {"cuda", "mps", "cpu"}


def test_unknown_device_rejected() -> None:
    with pytest.raises(DeviceError):
        resolve_device("tpu")


def test_missing_accelerator_rejected() -> None:
    if not torch.cuda.is_available():
        with pytest.raises(DeviceError):
            resolve_device("cuda")
        with pytest.raises(DeviceError):
            resolve_device("rocm")
    backend = getattr(torch.backends, "mps", None)
    if backend is None or not backend.is_available():
        with pytest.raises(DeviceError):
            resolve_device("mps")
