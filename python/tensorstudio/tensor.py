"""Tensor creation helpers and the public Tensor class."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _C

Tensor = _C.Tensor


def _with_requires_grad(value: Tensor, requires_grad: bool) -> Tensor:
    if requires_grad:
        value.requires_grad = True
    return value


def tensor(data: Any, dtype: str | None = None, requires_grad: bool = False) -> Tensor:
    """Create a Tensor from Python numeric data or a NumPy array."""

    return _C.tensor(data, dtype, requires_grad)


def from_numpy(array: np.ndarray, requires_grad: bool = False) -> Tensor:
    """Create a Tensor copy from a NumPy array."""

    return _C.from_numpy(np.asarray(array), requires_grad)


def zeros(
    shape: int | tuple[int, ...] | list[int],
    dtype: str = "float32",
    requires_grad: bool = False,
) -> Tensor:
    """Create a tensor filled with zeros."""

    return _with_requires_grad(_C.zeros(shape, dtype), requires_grad)


def zeros_like(
    input: Tensor,
    dtype: str | None = None,
    requires_grad: bool | None = None,
) -> Tensor:
    """Create a zero tensor with metadata copied from another tensor."""

    return zeros(
        input.shape,
        dtype=dtype or input.dtype,
        requires_grad=input.requires_grad if requires_grad is None else requires_grad,
    )


def empty(
    shape: int | tuple[int, ...] | list[int],
    dtype: str = "float32",
    requires_grad: bool = False,
) -> Tensor:
    """Create an uninitialized tensor.

    The current CPU storage implementation may contain zeroed memory; callers
    should not rely on any particular initial value.
    """

    return _with_requires_grad(_C.empty(shape, dtype), requires_grad)


def empty_like(
    input: Tensor,
    dtype: str | None = None,
    requires_grad: bool | None = None,
) -> Tensor:
    """Create an uninitialized tensor with metadata copied from another tensor."""

    return empty(
        input.shape,
        dtype=dtype or input.dtype,
        requires_grad=input.requires_grad if requires_grad is None else requires_grad,
    )


def ones(
    shape: int | tuple[int, ...] | list[int],
    dtype: str = "float32",
    requires_grad: bool = False,
) -> Tensor:
    """Create a tensor filled with ones."""

    return _with_requires_grad(_C.ones(shape, dtype), requires_grad)


def ones_like(
    input: Tensor,
    dtype: str | None = None,
    requires_grad: bool | None = None,
) -> Tensor:
    """Create a ones tensor with metadata copied from another tensor."""

    return ones(
        input.shape,
        dtype=dtype or input.dtype,
        requires_grad=input.requires_grad if requires_grad is None else requires_grad,
    )


def full(
    shape: int | tuple[int, ...] | list[int],
    fill_value: float,
    dtype: str = "float32",
    requires_grad: bool = False,
) -> Tensor:
    """Create a tensor filled with a scalar value."""

    return _with_requires_grad(_C.full(shape, fill_value, dtype), requires_grad)


def full_like(
    input: Tensor,
    fill_value: float,
    dtype: str | None = None,
    requires_grad: bool | None = None,
) -> Tensor:
    """Create a filled tensor with metadata copied from another tensor."""

    return full(
        input.shape,
        fill_value,
        dtype=dtype or input.dtype,
        requires_grad=input.requires_grad if requires_grad is None else requires_grad,
    )


def rand(
    shape: int | tuple[int, ...] | list[int],
    dtype: str = "float32",
    seed: int | None = None,
    requires_grad: bool = False,
) -> Tensor:
    """Create a tensor with uniform random values in [0, 1)."""

    return _with_requires_grad(_C.rand(shape, dtype, seed), requires_grad)


def rand_like(
    input: Tensor,
    dtype: str | None = None,
    seed: int | None = None,
    requires_grad: bool | None = None,
) -> Tensor:
    """Create a uniform random tensor with metadata copied from another tensor."""

    return rand(
        input.shape,
        dtype=dtype or input.dtype,
        seed=seed,
        requires_grad=input.requires_grad if requires_grad is None else requires_grad,
    )


def uniform(
    shape: int | tuple[int, ...] | list[int],
    low: float = 0.0,
    high: float = 1.0,
    dtype: str = "float32",
    seed: int | None = None,
    requires_grad: bool = False,
) -> Tensor:
    """Create a tensor sampled from ``Uniform(low, high)``."""

    return _with_requires_grad(_C.uniform(shape, low, high, dtype, seed), requires_grad)


def uniform_like(
    input: Tensor,
    low: float = 0.0,
    high: float = 1.0,
    dtype: str | None = None,
    seed: int | None = None,
    requires_grad: bool | None = None,
) -> Tensor:
    """Create a uniform random tensor with metadata copied from another tensor."""

    return uniform(
        input.shape,
        low=low,
        high=high,
        dtype=dtype or input.dtype,
        seed=seed,
        requires_grad=input.requires_grad if requires_grad is None else requires_grad,
    )


def randn(
    shape: int | tuple[int, ...] | list[int],
    dtype: str = "float32",
    seed: int | None = None,
    requires_grad: bool = False,
) -> Tensor:
    """Create a tensor with normally distributed pseudo-random values."""

    return _with_requires_grad(_C.randn(shape, dtype, seed), requires_grad)


def randn_like(
    input: Tensor,
    dtype: str | None = None,
    seed: int | None = None,
    requires_grad: bool | None = None,
) -> Tensor:
    """Create a normal random tensor with metadata copied from another tensor."""

    return randn(
        input.shape,
        dtype=dtype or input.dtype,
        seed=seed,
        requires_grad=input.requires_grad if requires_grad is None else requires_grad,
    )


def normal(
    shape: int | tuple[int, ...] | list[int],
    mean: float = 0.0,
    stddev: float = 1.0,
    dtype: str = "float32",
    seed: int | None = None,
    requires_grad: bool = False,
) -> Tensor:
    """Create a tensor sampled from ``Normal(mean, stddev)``."""

    return _with_requires_grad(_C.normal(shape, mean, stddev, dtype, seed), requires_grad)


def normal_like(
    input: Tensor,
    mean: float = 0.0,
    stddev: float = 1.0,
    dtype: str | None = None,
    seed: int | None = None,
    requires_grad: bool | None = None,
) -> Tensor:
    """Create a normal random tensor with metadata copied from another tensor."""

    return normal(
        input.shape,
        mean=mean,
        stddev=stddev,
        dtype=dtype or input.dtype,
        seed=seed,
        requires_grad=input.requires_grad if requires_grad is None else requires_grad,
    )


def randint(
    shape: int | tuple[int, ...] | list[int],
    low: int,
    high: int,
    dtype: str = "int64",
    seed: int | None = None,
) -> Tensor:
    """Create an integer tensor sampled uniformly from ``[low, high)``."""

    return _C.randint(shape, low, high, dtype, seed)


def randint_like(
    input: Tensor,
    low: int,
    high: int,
    dtype: str | None = None,
    seed: int | None = None,
) -> Tensor:
    """Create an integer random tensor with shape copied from another tensor."""

    return randint(input.shape, low=low, high=high, dtype=dtype or "int64", seed=seed)


def bernoulli(
    shape: int | tuple[int, ...] | list[int],
    probability: float = 0.5,
    dtype: str = "bool",
    seed: int | None = None,
) -> Tensor:
    """Create a tensor sampled from a Bernoulli distribution."""

    return _C.bernoulli(shape, probability, dtype, seed)


def bernoulli_like(
    input: Tensor,
    probability: float = 0.5,
    dtype: str = "bool",
    seed: int | None = None,
) -> Tensor:
    """Create a Bernoulli random tensor with shape copied from another tensor."""

    return bernoulli(input.shape, probability=probability, dtype=dtype, seed=seed)


def arange(
    start: float,
    stop: float | None = None,
    step: float = 1,
    dtype: str = "float32",
    requires_grad: bool = False,
) -> Tensor:
    """Create a 1D tensor with evenly spaced values."""

    return _with_requires_grad(_C.arange(start, stop, step, dtype), requires_grad)


def eye(
    n: int,
    m: int | None = None,
    dtype: str = "float32",
    requires_grad: bool = False,
) -> Tensor:
    """Create a 2D identity matrix."""

    return _with_requires_grad(_C.eye(n, m, dtype), requires_grad)


def linspace(
    start: float,
    stop: float,
    steps: int,
    dtype: str = "float32",
    requires_grad: bool = False,
) -> Tensor:
    """Create a 1D tensor with evenly spaced values including both endpoints."""

    return _with_requires_grad(_C.linspace(start, stop, steps, dtype), requires_grad)


def manual_seed(seed: int) -> None:
    """Seed TensorStudio's process-local random generator."""

    _C.manual_seed(seed)


__all__ = [
    "Tensor",
    "arange",
    "bernoulli",
    "bernoulli_like",
    "empty",
    "empty_like",
    "eye",
    "from_numpy",
    "full",
    "full_like",
    "linspace",
    "manual_seed",
    "normal",
    "normal_like",
    "ones",
    "ones_like",
    "rand",
    "rand_like",
    "randint",
    "randint_like",
    "randn",
    "randn_like",
    "tensor",
    "uniform",
    "uniform_like",
    "zeros",
    "zeros_like",
]
