"""Higher-level tensor math helpers built from native TensorStudio ops."""

from __future__ import annotations

from typing import Literal

from .tensor import Tensor

NormOrder = Literal[1, 2, "fro", "inf"]


def square(input: Tensor) -> Tensor:
    """Return ``input * input`` with autograd support."""

    return input * input


def reciprocal(input: Tensor) -> Tensor:
    """Return the elementwise reciprocal ``1 / input``."""

    return 1.0 / input


def variance(
    input: Tensor,
    axis: int | None = None,
    keepdims: bool = False,
    correction: int = 0,
) -> Tensor:
    """Compute variance over all elements or one axis.

    ``correction=1`` gives the usual sample variance denominator. TensorStudio
    currently supports a single optional axis, matching the C++ reduction API.
    """

    if correction < 0:
        raise ValueError("correction must be non-negative")
    if axis is None:
        denom = input.size - correction
    else:
        normalized_axis = axis if axis >= 0 else input.ndim + axis
        if normalized_axis < 0 or normalized_axis >= input.ndim:
            raise ValueError("axis is out of range")
        denom = input.shape[normalized_axis] - correction
    if denom <= 0:
        raise ValueError("variance denominator must be positive")
    mean = input.mean(axis=axis, keepdims=True)
    centered = input - mean
    reduced = (centered * centered).sum(axis=axis, keepdims=keepdims)
    return reduced / float(denom)


def std(
    input: Tensor,
    axis: int | None = None,
    keepdims: bool = False,
    correction: int = 0,
) -> Tensor:
    """Compute standard deviation over all elements or one axis."""

    return variance(input, axis=axis, keepdims=keepdims, correction=correction).sqrt()


def norm(
    input: Tensor,
    ord: NormOrder = 2,
    axis: int | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Compute a vector-style norm over all elements or one axis."""

    magnitude = input.abs()
    if ord == 1:
        return magnitude.sum(axis=axis, keepdims=keepdims)
    if ord == 2 or ord == "fro":
        return (input * input).sum(axis=axis, keepdims=keepdims).sqrt()
    if ord == "inf":
        return magnitude.max(axis=axis, keepdims=keepdims)
    raise ValueError("ord must be 1, 2, 'fro', or 'inf'")


__all__ = ["NormOrder", "norm", "reciprocal", "square", "std", "variance"]
