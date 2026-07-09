"""Hardware backend and device helpers."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

from . import _C

Device = _C.Device
DeviceLike = str | Device
_state = threading.local()


def device(spec: DeviceLike = "cpu") -> Device:
    """Create a TensorStudio device descriptor."""

    if isinstance(spec, Device):
        return spec
    return _C.device(spec)


def _device_stack() -> list[Device]:
    stack = getattr(_state, "device_stack", None)
    if stack is None:
        stack = []
        _state.device_stack = stack
    return cast(list[Device], stack)


def current_device() -> Device:
    """Return the current thread-local default placement device."""

    stack = _device_stack()
    if stack:
        return stack[-1]
    selected = getattr(_state, "default_device", None)
    if selected is None:
        return device("cpu")
    return cast(Device, selected)


def set_default_device(spec: DeviceLike = "cpu") -> Device:
    """Set the thread-local default placement device and return it."""

    selected = device(spec)
    _state.default_device = selected
    return selected


def reset_default_device() -> None:
    """Reset the thread-local default placement device to CPU."""

    if hasattr(_state, "default_device"):
        delattr(_state, "default_device")


@contextmanager
def device_scope(spec: DeviceLike) -> Iterator[Device]:
    """Temporarily set the current thread-local placement device."""

    selected = device(spec)
    stack = _device_stack()
    stack.append(selected)
    try:
        yield selected
    finally:
        stack.pop()


def is_available(spec: DeviceLike = "cpu") -> bool:
    """Return whether a concrete device is available in this build."""

    return bool(_C.is_device_available(device(spec)))


def device_count(spec: DeviceLike = "cpu") -> int:
    """Return the number of visible devices for a backend descriptor."""

    return int(_C.device_count(device(spec)))


def available_devices() -> list[Device]:
    """Return available concrete devices."""

    return list(_C.available_devices())


def backend_info() -> list[dict[str, Any]]:
    """Return backend availability and build metadata."""

    return list(_C.backend_info())


def backend_allocator_info(spec: DeviceLike | None = None) -> list[dict[str, Any]]:
    """Return allocator and memory-space metadata for one backend or all backends."""

    selected = None if spec is None else device(spec)
    return list(_C.backend_allocator_info(selected))


def storage_telemetry() -> dict[str, int]:
    """Return native CPU storage allocation counters for diagnostics."""

    raw = cast(dict[str, int], dict(_C.storage_telemetry()))
    return {str(key): int(value) for key, value in raw.items()}


def reset_storage_telemetry() -> None:
    """Reset cumulative storage counters while preserving current active usage."""

    _C.reset_storage_telemetry()


def backend_runtime_info(spec: DeviceLike | None = None) -> list[dict[str, Any]]:
    """Return runtime, synchronization, and compiler-boundary metadata."""

    selected = None if spec is None else device(spec)
    return list(_C.backend_runtime_info(selected))


def backend_device_properties(spec: DeviceLike | None = None) -> list[dict[str, Any]]:
    """Return TensorFlow-style physical device property metadata."""

    selected = None if spec is None else device(spec)
    return list(_C.backend_device_properties(selected))


def logical_device_info(spec: DeviceLike | None = None) -> list[dict[str, Any]]:
    """Return logical devices exposed by available physical devices."""

    selected = None if spec is None else device(spec)
    return list(_C.logical_device_info(selected))


def backend_op_info(op: str | None = None) -> list[dict[str, Any]] | dict[str, Any]:
    """Return registered operation-interface metadata."""

    value = _C.backend_op_info(op)
    if op is None:
        return list(cast(list[dict[str, Any]], value))
    return dict(cast(dict[str, Any], value))


def backend_kernel_info(spec: DeviceLike | None = None) -> list[dict[str, Any]]:
    """Return registered backend kernel capability metadata."""

    selected = None if spec is None else device(spec)
    return list(_C.backend_kernel_info(selected))


def backend_supports_kernel(
    op: str,
    spec: DeviceLike = "cpu",
    dtype: str = "float32",
) -> bool:
    """Return whether a backend has a registered kernel for an op and dtype."""

    return bool(_C.backend_supports_kernel(op, device(spec), dtype))


def kernel_placement_info(
    op: str,
    spec: DeviceLike = "cpu",
    dtype: str = "float32",
) -> dict[str, Any]:
    """Return explicit placement support metadata for a requested backend."""

    return dict(_C.kernel_placement_info(op, device(spec), dtype))


def backend_execution_plan(
    op: str,
    spec: DeviceLike = "cpu",
    dtype: str = "float32",
    input_devices: DeviceLike | list[DeviceLike] | tuple[DeviceLike, ...] | None = None,
) -> dict[str, Any]:
    """Return an explicit eager execution/fallback/transfer plan for an op."""

    normalized_inputs: object
    if input_devices is None:
        normalized_inputs = None
    elif isinstance(input_devices, (str, Device)):
        normalized_inputs = device(input_devices)
    else:
        normalized_inputs = [device(item) for item in input_devices]
    return dict(_C.backend_execution_plan(op, device(spec), dtype, normalized_inputs))


def device_transfer_info(source: DeviceLike, target: DeviceLike) -> dict[str, Any]:
    """Return transfer support metadata for a source/target device pair."""

    return dict(_C.device_transfer_info(device(source), device(target)))


def can_transfer(source: DeviceLike, target: DeviceLike) -> bool:
    """Return whether TensorStudio can transfer tensors between two devices."""

    return bool(_C.can_transfer(device(source), device(target)))


def register_backend(
    name: str,
    *,
    compiled: bool = True,
    available: bool = False,
    device_count: int = 0,
    reason: str = "",
) -> None:
    """Register an external backend descriptor for diagnostics and planning."""

    _C.register_backend(name, compiled, available, device_count, reason)


def unregister_backend(name: str) -> None:
    """Unregister an external backend descriptor and its kernel metadata."""

    _C.unregister_backend(name)


def register_backend_kernel(
    backend: str,
    op: str,
    dtypes: list[str] | tuple[str, ...],
    *,
    forward: bool = True,
    backward: bool = False,
    available: bool = True,
    reason: str = "",
    memory_space: str = "device",
    execution_mode: str = "sync",
    deterministic: bool = True,
    priority: int = 50,
) -> None:
    """Register external backend kernel metadata without loading code."""

    _C.register_backend_kernel(
        backend,
        op,
        list(dtypes),
        forward,
        backward,
        available,
        reason,
        memory_space,
        execution_mode,
        deterministic,
        priority,
    )


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
    "backend_allocator_info",
    "available_devices",
    "backend_device_properties",
    "backend_execution_plan",
    "backend_info",
    "backend_kernel_info",
    "backend_op_info",
    "backend_runtime_info",
    "backend_supports_kernel",
    "can_transfer",
    "current_device",
    "cuda_is_available",
    "device",
    "device_count",
    "device_scope",
    "device_transfer_info",
    "is_available",
    "kernel_placement_info",
    "logical_device_info",
    "metal_is_available",
    "register_backend",
    "register_backend_kernel",
    "require_device",
    "reset_default_device",
    "reset_storage_telemetry",
    "set_default_device",
    "storage_telemetry",
    "unregister_backend",
]
