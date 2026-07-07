"""Higher-level tensor math helpers built from native TensorStudio ops."""

from __future__ import annotations

from typing import Literal

from . import _C
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
    if axis is None or isinstance(axis, int):
        return _C.variance(input, axis, keepdims, correction)
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

    if axis is None or isinstance(axis, int):
        return _C.std(input, axis, keepdims, correction)
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


def logsumexp(input: Tensor, axis: AxisLike = None, keepdims: bool = False) -> Tensor:
    """Compute a numerically stable log(sum(exp(input)))."""

    return _C.logsumexp(input, axis, keepdims)


def softmax(input: Tensor, axis: int = -1) -> Tensor:
    """Compute softmax along one axis with max-shifted stable numerics."""

    return _C.softmax(input, axis)


def log_softmax(input: Tensor, axis: int = -1) -> Tensor:
    """Compute log-softmax along one axis with max-shifted stable numerics."""

    return _C.log_softmax(input, axis)


def all(input: Tensor, axis: AxisLike = None, keepdims: bool = False) -> Tensor:  # noqa: A001
    """Return whether all selected values are non-zero."""

    return _C.all(input, axis, keepdims)


def any(input: Tensor, axis: AxisLike = None, keepdims: bool = False) -> Tensor:  # noqa: A001
    """Return whether any selected value is non-zero."""

    return _C.any(input, axis, keepdims)


def einsum(pattern: str, *operands: Tensor) -> Tensor:
    """Evaluate a practical subset of Einstein summation patterns.

    Supported in v1.7.0:
    ``ij,jk->ik``, ``bi,ij->bj``, ``bij,bjk->bik``, ``i,i->``,
    ``ij,ij->``, ``ij->ji``, ``bij->bji``, ``i->``, ``ij->``,
    ``ij->i``, and ``ij->j``.
    """

    equation = pattern.replace(" ", "")
    if "->" not in equation:
        raise ValueError("einsum pattern must include '->'")
    lhs, output = equation.split("->", 1)
    inputs = lhs.split(",") if lhs else []
    if len(inputs) != len(operands):
        raise ValueError("einsum operand count does not match the pattern")

    if len(operands) == 2:
        left, right = operands
        left_spec, right_spec = inputs
        if (left_spec, right_spec, output) == ("ij", "jk", "ik"):
            return left @ right
        if (left_spec, right_spec, output) == ("bi", "ij", "bj"):
            return left @ right
        if (left_spec, right_spec, output) == ("bij", "bjk", "bik"):
            return left @ right
        if (left_spec, right_spec, output) in {("i", "i", ""), ("ij", "ij", "")}:
            return (left * right).sum()

    if len(operands) == 1:
        (value,) = operands
        (spec,) = inputs
        if (spec, output) == ("ij", "ji"):
            return value.T
        if (spec, output) == ("bij", "bji"):
            return value.transpose(1, 2)
        if (spec, output) in {("i", ""), ("ij", "")}:
            return value.sum()
        if (spec, output) == ("ij", "i"):
            return value.sum(axis=1)
        if (spec, output) == ("ij", "j"):
            return value.sum(axis=0)

    raise ValueError(f"unsupported TensorStudio einsum pattern: {pattern!r}")


__all__ = [
    "AxisLike",
    "NormOrder",
    "all",
    "any",
    "einsum",
    "log_softmax",
    "logsumexp",
    "norm",
    "reciprocal",
    "square",
    "softmax",
    "std",
    "variance",
]
