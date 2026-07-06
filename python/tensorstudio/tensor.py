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


def empty(shape: int | tuple[int, ...] | list[int], dtype: str = "float32") -> Tensor:
    """Create an uninitialized tensor.

    The current CPU storage implementation may contain zeroed memory; callers
    should not rely on any particular initial value.
    """

    return _C.empty(shape, dtype)


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


def rand(
    shape: int | tuple[int, ...] | list[int],
    dtype: str = "float32",
    seed: int | None = None,
) -> Tensor:
    """Create a tensor with uniform random values in [0, 1)."""

    return _C.rand(shape, dtype, seed)


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


def eye(n: int, m: int | None = None, dtype: str = "float32") -> Tensor:
    """Create a 2D identity matrix."""

    return _C.eye(n, m, dtype)


def linspace(start: float, stop: float, steps: int, dtype: str = "float32") -> Tensor:
    """Create a 1D tensor with evenly spaced values including both endpoints."""

    return _C.linspace(start, stop, steps, dtype)


def manual_seed(seed: int) -> None:
    """Seed TensorStudio's process-local random generator."""

    _C.manual_seed(seed)


__all__ = [
    "Tensor",
    "arange",
    "empty",
    "eye",
    "from_numpy",
    "full",
    "linspace",
    "manual_seed",
    "ones",
    "rand",
    "randn",
    "tensor",
    "zeros",
]
