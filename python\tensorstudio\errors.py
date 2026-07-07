"""TensorStudio exception aliases."""

from __future__ import annotations

from . import _C

TensorStudioError = _C.TensorStudioError
ShapeError = _C.ShapeError
DTypeError = _C.DTypeError
DeviceError = _C.DeviceError
AutogradError = _C.AutogradError

__all__ = [
    "AutogradError",
    "DTypeError",
    "DeviceError",
    "ShapeError",
    "TensorStudioError",
]
