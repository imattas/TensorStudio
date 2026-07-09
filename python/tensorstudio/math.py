"""Higher-level tensor math helpers built from native TensorStudio ops."""

from __future__ import annotations

from typing import Literal

from .tensor import Tensor

NormOrder = Literal[1, 2, "fro", "inf"]
AxisLike = int | tuple[int, ...] | list[int] | None


def _normalized_axes(axis: AxisLike, ndim: int) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    axes = (axis,) if isinstance(axis, int) else tuple(axis)

    normalized: list[int] = []
    seen: set[int] = set()
    for value in axes:
        normalized_axis = value if value >= 0 else ndim + value
        if normalized_axis < 0 or normalized_axis >= ndim:
            raise ValueError("axis is out of range")
        if normalized_axis in seen:
            raise ValueError("duplicate reduction axis")
        seen.add(normalized_axis)
        normalized.append(normalized_axis)
    return tuple(normalized)


def square(input: Tensor) -> Tensor:
    """Return ``input * input`` with autograd support."""

    return input * input


def reciprocal(input: Tensor) -> Tensor:
    """Return the elementwise reciprocal ``1 / input``."""

    return 1.0 / input


def variance(
    input: Tensor,
    axis: AxisLike = None,
    keepdims: bool = False,
    correction: int = 0,
) -> Tensor:
    """Compute variance over all elements, one axis, or a tuple/list of axes.

    ``correction=1`` gives the usual sample variance denominator. TensorStudio
    uses the product of the reduced axes as the denominator base.
    """

    if correction < 0:
        raise ValueError("correction must be non-negative")
    axes = _normalized_axes(axis, input.ndim)
    denom = 1
    for normalized_axis in axes:
        denom *= input.shape[normalized_axis]
    denom -= correction
    if denom <= 0:
        raise ValueError("variance denominator must be positive")
    mean = input.mean(axis=axis, keepdims=True)
    centered = input - mean
    reduced = (centered * centered).sum(axis=axis, keepdims=keepdims)
    return reduced / float(denom)


def std(
    input: Tensor,
    axis: AxisLike = None,
    keepdims: bool = False,
    correction: int = 0,
) -> Tensor:
    """Compute standard deviation over all elements, one axis, or a tuple/list of axes."""

    return variance(input, axis=axis, keepdims=keepdims, correction=correction).sqrt()


def norm(
    input: Tensor,
    ord: NormOrder = 2,
    axis: AxisLike = None,
    keepdims: bool = False,
) -> Tensor:
    """Compute a vector-style norm over all elements, one axis, or a tuple/list of axes."""

    magnitude = input.abs()
    if ord == 1:
        return magnitude.sum(axis=axis, keepdims=keepdims)
    if ord == 2 or ord == "fro":
        return (input * input).sum(axis=axis, keepdims=keepdims).sqrt()
    if ord == "inf":
        return magnitude.max(axis=axis, keepdims=keepdims)
    raise ValueError("ord must be 1, 2, 'fro', or 'inf'")


__all__ = [
    "AxisLike",
    "NormOrder",
    "norm",
    "reciprocal",
    "square",
    "std",
    "variance",
]
