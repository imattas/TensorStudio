"""Tensor creation helpers and the public Tensor class."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _C

Tensor = _C.Tensor


def tensor(data: Any, dtype: str | None = None, requires_grad: bool = False) -> Tensor:
    """Create a Tensor from Python numeric data or a NumPy array."""

    return _C.tensor(data, dtype, requires_grad)


def from_numpy(array: np.ndarray, requires_grad: bool = False) -> Tensor:
    """Create a Tensor copy from a NumPy array."""

    return _C.from_numpy(np.asarray(array), requires_grad)


def zeros(shape: int | tuple[int, ...] | list[int], dtype: str = "float32") -> Tensor:
    """Create a tensor filled with zeros."""

    return _C.zeros(shape, dtype)


def ones(shape: int | tuple[int, ...] | list[int], dtype: str = "float32") -> Tensor:
    """Create a tensor filled with ones."""

    return _C.ones(shape, dtype)


def full(
    shape: int | tuple[int, ...] | list[int],
    fill_value: float,
    dtype: str = "float32",
) -> Tensor:
    """Create a tensor filled with a scalar value."""

    return _C.full(shape, fill_value, dtype)


def randn(
    shape: int | tuple[int, ...] | list[int],
    dtype: str = "float32",
    seed: int | None = None,
) -> Tensor:
    """Create a tensor with normally distributed pseudo-random values."""

    return _C.randn(shape, dtype, seed)


def arange(
    start: float,
    stop: float | None = None,
    step: float = 1,
    dtype: str = "float32",
) -> Tensor:
    """Create a 1D tensor with evenly spaced values."""

    return _C.arange(start, stop, step, dtype)


__all__ = ["Tensor", "arange", "from_numpy", "full", "ones", "randn", "tensor", "zeros"]
