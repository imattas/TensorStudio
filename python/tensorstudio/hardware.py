"""Hardware backend and device helpers."""

from __future__ import annotations

from typing import Any

from . import _C

Device = _C.Device
DeviceLike = str | Device


def device(spec: DeviceLike = "cpu") -> Device:
    """Create a TensorStudio device descriptor."""

    if isinstance(spec, Device):
        return spec
    return _C.device(spec)


def is_available(spec: DeviceLike = "cpu") -> bool:
    """Return whether a concrete device is available in this build."""

    return bool(_C.is_device_available(device(spec)))


def device_count(kind: str = "cpu") -> int:
    """Return the number of visible devices for a backend kind."""

    return int(_C.device_count(kind))


def available_devices() -> list[Device]:
    """Return available concrete devices."""

    return list(_C.available_devices())


def backend_info() -> list[dict[str, Any]]:
    """Return backend availability and build metadata."""

    return list(_C.backend_info())


def cuda_is_available() -> bool:
    """Return whether CUDA tensors can be created in this build."""

    return is_available("cuda")


def metal_is_available() -> bool:
    """Return whether Metal tensors can be created in this build."""

    return is_available("metal")


def require_device(spec: DeviceLike) -> Device:
    """Return a device or raise DeviceError if it is unavailable."""

    selected = device(spec)
    if not is_available(selected):
        raise _C.DeviceError(f"device {selected} is not available in this TensorStudio build")
    return selected


__all__ = [
    "Device",
    "DeviceLike",
    "available_devices",
    "backend_info",
    "cuda_is_available",
    "device",
    "device_count",
    "is_available",
    "metal_is_available",
    "require_device",
]
