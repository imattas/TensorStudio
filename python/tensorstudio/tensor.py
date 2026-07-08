"""Tensor creation helpers and the public Tensor class."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _C
from .hardware import DeviceLike
from .hardware import device as make_device

Tensor = _C.Tensor


def _finish_tensor(
    value: Tensor,
    requires_grad: bool = False,
    target_device: DeviceLike = "cpu",
) -> Tensor:
    if requires_grad:
        value.requires_grad = True
    return value.to_device(make_device(target_device))


def tensor(
    data: Any,
    dtype: str | None = None,
    requires_grad: bool = False,
    device: DeviceLike = "cpu",
) -> Tensor:
    """Create a Tensor from Python numeric data or a NumPy array."""

    return _finish_tensor(_C.tensor(data, dtype, requires_grad), target_device=device)


def from_numpy(
    array: np.ndarray,
    requires_grad: bool = False,
    device: DeviceLike = "cpu",
) -> Tensor:
    """Create a Tensor copy from a NumPy array."""

    return _finish_tensor(_C.from_numpy(np.asarray(array), requires_grad), target_device=device)


def from_dlpack(
    source: Any,
    requires_grad: bool = False,
    device: DeviceLike = "cpu",
) -> Tensor:
    """Create a Tensor copy from a CPU DLPack-compatible object."""

    if not hasattr(source, "__dlpack__"):
        raise TypeError("from_dlpack expects an object implementing __dlpack__")
    array = np.from_dlpack(source)
    return from_numpy(np.asarray(array), requires_grad=requires_grad, device=device)


def zeros(
    shape: int | tuple[int, ...] | list[int],
    dtype: str = "float32",
    requires_grad: bool = False,
    device: DeviceLike = "cpu",
) -> Tensor:
    """Create a tensor filled with zeros."""

    return _finish_tensor(_C.zeros(shape, dtype), requires_grad, device)


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
        device=input.device,
    )


def empty(
    shape: int | tuple[int, ...] | list[int],
    dtype: str = "float32",
    requires_grad: bool = False,
    device: DeviceLike = "cpu",
) -> Tensor:
    """Create an uninitialized tensor.

    The current CPU storage implementation may contain zeroed memory; callers
    should not rely on any particular initial value.
    """

    return _finish_tensor(_C.empty(shape, dtype), requires_grad, device)


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
        device=input.device,
    )


def ones(
    shape: int | tuple[int, ...] | list[int],
    dtype: str = "float32",
    requires_grad: bool = False,
    device: DeviceLike = "cpu",
) -> Tensor:
    """Create a tensor filled with ones."""

    return _finish_tensor(_C.ones(shape, dtype), requires_grad, device)


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
        device=input.device,
    )


def full(
    shape: int | tuple[int, ...] | list[int],
    fill_value: float,
    dtype: str = "float32",
    requires_grad: bool = False,
    device: DeviceLike = "cpu",
) -> Tensor:
    """Create a tensor filled with a scalar value."""

    return _finish_tensor(_C.full(shape, fill_value, dtype), requires_grad, device)


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
        device=input.device,
    )


def rand(
    shape: int | tuple[int, ...] | list[int],
    dtype: str = "float32",
    seed: int | None = None,
    requires_grad: bool = False,
    device: DeviceLike = "cpu",
) -> Tensor:
    """Create a tensor with uniform random values in [0, 1)."""

    return _finish_tensor(_C.rand(shape, dtype, seed), requires_grad, device)


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
        device=input.device,
    )


def uniform(
    shape: int | tuple[int, ...] | list[int],
    low: float = 0.0,
    high: float = 1.0,
    dtype: str = "float32",
    seed: int | None = None,
    requires_grad: bool = False,
    device: DeviceLike = "cpu",
) -> Tensor:
    """Create a tensor sampled from ``Uniform(low, high)``."""

    return _finish_tensor(_C.uniform(shape, low, high, dtype, seed), requires_grad, device)


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
        device=input.device,
    )


def randn(
    shape: int | tuple[int, ...] | list[int],
    dtype: str = "float32",
    seed: int | None = None,
    requires_grad: bool = False,
    device: DeviceLike = "cpu",
) -> Tensor:
    """Create a tensor with normally distributed pseudo-random values."""

    return _finish_tensor(_C.randn(shape, dtype, seed), requires_grad, device)


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
        device=input.device,
    )


def normal(
    shape: int | tuple[int, ...] | list[int],
    mean: float = 0.0,
    stddev: float = 1.0,
    dtype: str = "float32",
    seed: int | None = None,
    requires_grad: bool = False,
    device: DeviceLike = "cpu",
) -> Tensor:
    """Create a tensor sampled from ``Normal(mean, stddev)``."""

    return _finish_tensor(_C.normal(shape, mean, stddev, dtype, seed), requires_grad, device)


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
        device=input.device,
    )


def randint(
    shape: int | tuple[int, ...] | list[int],
    low: int,
    high: int,
    dtype: str = "int64",
    seed: int | None = None,
    device: DeviceLike = "cpu",
) -> Tensor:
    """Create an integer tensor sampled uniformly from ``[low, high)``."""

    return _finish_tensor(_C.randint(shape, low, high, dtype, seed), target_device=device)


def randint_like(
    input: Tensor,
    low: int,
    high: int,
    dtype: str | None = None,
    seed: int | None = None,
) -> Tensor:
    """Create an integer random tensor with shape copied from another tensor."""

    return randint(
        input.shape,
        low=low,
        high=high,
        dtype=dtype or "int64",
        seed=seed,
        device=input.device,
    )


def bernoulli(
    shape: int | tuple[int, ...] | list[int],
    probability: float = 0.5,
    dtype: str = "bool",
    seed: int | None = None,
    device: DeviceLike = "cpu",
) -> Tensor:
    """Create a tensor sampled from a Bernoulli distribution."""

    return _finish_tensor(_C.bernoulli(shape, probability, dtype, seed), target_device=device)


def bernoulli_like(
    input: Tensor,
    probability: float = 0.5,
    dtype: str = "bool",
    seed: int | None = None,
) -> Tensor:
    """Create a Bernoulli random tensor with shape copied from another tensor."""

    return bernoulli(
        input.shape,
        probability=probability,
        dtype=dtype,
        seed=seed,
        device=input.device,
    )


def arange(
    start: float,
    stop: float | None = None,
    step: float = 1,
    dtype: str = "float32",
    requires_grad: bool = False,
    device: DeviceLike = "cpu",
) -> Tensor:
    """Create a 1D tensor with evenly spaced values."""

    return _finish_tensor(_C.arange(start, stop, step, dtype), requires_grad, device)


def eye(
    n: int,
    m: int | None = None,
    dtype: str = "float32",
    requires_grad: bool = False,
    device: DeviceLike = "cpu",
) -> Tensor:
    """Create a 2D identity matrix."""

    return _finish_tensor(_C.eye(n, m, dtype), requires_grad, device)


def linspace(
    start: float,
    stop: float,
    steps: int,
    dtype: str = "float32",
    requires_grad: bool = False,
    device: DeviceLike = "cpu",
) -> Tensor:
    """Create a 1D tensor with evenly spaced values including both endpoints."""

    return _finish_tensor(_C.linspace(start, stop, steps, dtype), requires_grad, device)


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
    "from_dlpack",
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
