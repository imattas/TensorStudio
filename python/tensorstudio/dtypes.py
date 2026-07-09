"""DType normalization, promotion, and casting-policy helpers."""

from __future__ import annotations

from typing import Any, Literal
from typing import cast as type_cast

import numpy as np

from . import _C
from .tensor import Tensor, tensor

DTypeName = Literal["bool", "int32", "int64", "float32", "float64"]
CastingMode = Literal["no", "equiv", "safe", "same_kind", "unsafe"]
BinaryOpName = Literal[
    "add",
    "sub",
    "mul",
    "matmul",
    "div",
    "truediv",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
]

DTYPE_PROMOTION_ORDER: tuple[DTypeName, ...] = ("bool", "int32", "int64", "float32", "float64")
_VALID_DTYPES = set(DTYPE_PROMOTION_ORDER)
_COMPARISON_OPS = {"eq", "ne", "lt", "le", "gt", "ge"}
_ARITHMETIC_OPS = {"add", "sub", "mul", "matmul"}
_DIVISION_OPS = {"div", "truediv"}
_VALID_CASTING_MODES = {"no", "equiv", "safe", "same_kind", "unsafe"}
_DTYPE_KIND: dict[DTypeName, str] = {
    "bool": "bool",
    "int32": "signed_integer",
    "int64": "signed_integer",
    "float32": "floating",
    "float64": "floating",
}
_SAFE_CASTS: set[tuple[DTypeName, DTypeName]] = {
    ("bool", "bool"),
    ("bool", "int32"),
    ("bool", "int64"),
    ("bool", "float32"),
    ("bool", "float64"),
    ("int32", "int32"),
    ("int32", "int64"),
    ("int32", "float64"),
    ("int64", "int64"),
    ("float32", "float32"),
    ("float32", "float64"),
    ("float64", "float64"),
}


def _canonical_dtype(dtype: str) -> DTypeName:
    if dtype not in _VALID_DTYPES:
        raise ValueError(f"unsupported TensorStudio dtype: {dtype!r}")
    return dtype


def _casting_mode(casting: str) -> CastingMode:
    if casting not in _VALID_CASTING_MODES:
        raise ValueError(
            "casting must be one of 'no', 'equiv', 'safe', 'same_kind', or 'unsafe'"
        )
    return type_cast(CastingMode, casting)


def normalize_dtype(dtype: str) -> DTypeName:
    """Normalize a dtype alias to TensorStudio's canonical dtype name."""

    return _canonical_dtype(tensor([], dtype=dtype).dtype)


def dtype_of(value: Any) -> DTypeName:
    """Return TensorStudio's dtype name for a tensor, dtype string, scalar, or array."""

    if isinstance(value, Tensor):
        return _canonical_dtype(value.dtype)
    if isinstance(value, str):
        return normalize_dtype(value)
    if isinstance(value, np.ndarray):
        return normalize_dtype(str(value.dtype))
    return _canonical_dtype(tensor(value).dtype)


def promote_types(left: str, right: str) -> DTypeName:
    """Return the arithmetic promotion of two dtype names.

    This is the native C++ promotion rule used by `add`, `sub`, `mul`, and
    `matmul`.
    """

    return _canonical_dtype(_C.promote_types(left, right))


def result_type(left: Any, right: Any, op: BinaryOpName = "add") -> DTypeName:
    """Return the result dtype for a TensorStudio binary operation."""

    left_dtype = dtype_of(left)
    right_dtype = dtype_of(right)
    if op in _ARITHMETIC_OPS:
        return promote_types(left_dtype, right_dtype)
    if op in _DIVISION_OPS:
        return "float64" if "float64" in {left_dtype, right_dtype} else "float32"
    if op in _COMPARISON_OPS:
        return "bool"
    raise ValueError(f"unsupported binary op for result_type: {op!r}")


def can_cast(from_: Any, to: str, casting: CastingMode = "safe") -> bool:
    """Return whether TensorStudio permits a dtype cast under a casting mode.

    Modes:

    - `no` and `equiv`: only exact canonical dtype matches.
    - `safe`: only casts that preserve every value exactly for the source dtype.
    - `same_kind`: safe casts plus casts within integer or floating dtype families.
    - `unsafe`: any supported dtype conversion.
    """

    source = dtype_of(from_)
    target = normalize_dtype(to)
    mode = _casting_mode(casting)

    if mode in {"no", "equiv"}:
        return source == target
    if mode == "safe":
        return (source, target) in _SAFE_CASTS
    if mode == "same_kind":
        return (source, target) in _SAFE_CASTS or _DTYPE_KIND[source] == _DTYPE_KIND[target]
    return True


def cast(value: Any, dtype: str, casting: CastingMode = "unsafe") -> Tensor:
    """Cast a tensor-like value to `dtype` after checking the requested policy."""

    input = value if isinstance(value, Tensor) else tensor(value)
    target = normalize_dtype(dtype)
    mode = _casting_mode(casting)
    if not can_cast(input, target, mode):
        raise TypeError(
            f"cannot cast TensorStudio dtype {input.dtype!r} to {target!r} "
            f"under casting={mode!r}"
        )
    if input.dtype == target:
        return input
    return _C.astype(input, target)


def _install_tensor_cast_methods() -> None:
    def astype(self: Tensor, dtype: str, casting: CastingMode = "unsafe") -> Tensor:
        return cast(self, dtype, casting=casting)

    def to(self: Tensor, dtype: str, casting: CastingMode = "unsafe") -> Tensor:
        return cast(self, dtype, casting=casting)

    setattr(Tensor, "astype", astype)  # noqa: B010
    setattr(Tensor, "to", to)  # noqa: B010


_install_tensor_cast_methods()


__all__ = [
    "BinaryOpName",
    "CastingMode",
    "DTYPE_PROMOTION_ORDER",
    "DTypeName",
    "can_cast",
    "cast",
    "dtype_of",
    "normalize_dtype",
    "promote_types",
    "result_type",
]
