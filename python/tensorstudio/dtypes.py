"""DType normalization and binary operation result-type helpers."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from . import _C
from .tensor import Tensor, tensor

DTypeName = Literal["bool", "int32", "int64", "float32", "float64"]
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


def _canonical_dtype(dtype: str) -> DTypeName:
    if dtype not in _VALID_DTYPES:
        raise ValueError(f"unsupported TensorStudio dtype: {dtype!r}")
    return dtype


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


__all__ = [
    "BinaryOpName",
    "DTYPE_PROMOTION_ORDER",
    "DTypeName",
    "dtype_of",
    "normalize_dtype",
    "promote_types",
    "result_type",
]
