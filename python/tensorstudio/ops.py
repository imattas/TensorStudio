"""Functional tensor operations."""

from __future__ import annotations

from typing import Any

from . import _C
from .tensor import Tensor

PairLike = int | tuple[int, int] | list[int]
AxisLike = int | tuple[int, ...] | list[int] | None


def _pair(value: PairLike, name: str) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    if len(value) != 2:
        raise ValueError(f"{name} must be an int or a pair of ints")
    left, right = int(value[0]), int(value[1])
    return (left, right)


def add(left: Tensor, right: Any) -> Tensor:
    return _C.add(left, right)


def sub(left: Tensor, right: Any) -> Tensor:
    return _C.sub(left, right)


def mul(left: Tensor, right: Any) -> Tensor:
    return _C.mul(left, right)


def div(left: Tensor, right: Any) -> Tensor:
    return _C.div(left, right)


def equal(left: Tensor, right: Any) -> Tensor:
    return _C.equal(left, right)


def not_equal(left: Tensor, right: Any) -> Tensor:
    return _C.not_equal(left, right)


def less(left: Tensor, right: Any) -> Tensor:
    return _C.less(left, right)


def less_equal(left: Tensor, right: Any) -> Tensor:
    return _C.less_equal(left, right)


def greater(left: Tensor, right: Any) -> Tensor:
    return _C.greater(left, right)


def greater_equal(left: Tensor, right: Any) -> Tensor:
    return _C.greater_equal(left, right)


def maximum(left: Tensor, right: Any) -> Tensor:
    return _C.maximum(left, right)


def minimum(left: Tensor, right: Any) -> Tensor:
    return _C.minimum(left, right)


def where(condition: Any, true_value: Any, false_value: Any) -> Tensor:
    return _C.where(condition, true_value, false_value)


def neg(input: Tensor) -> Tensor:
    return _C.neg(input)


def pow(input: Tensor, exponent: float) -> Tensor:
    return _C.pow(input, exponent)


def matmul(left: Tensor, right: Any) -> Tensor:
    return _C.matmul(left, right)


def bmm(left: Tensor, right: Any) -> Tensor:
    return _C.bmm(left, right)


def conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: PairLike = 1,
    padding: PairLike = 0,
    dilation: PairLike = 1,
    groups: int = 1,
) -> Tensor:
    """Apply a 2D NCHW convolution with CPU kernels and autograd support."""

    stride_h, stride_w = _pair(stride, "stride")
    padding_h, padding_w = _pair(padding, "padding")
    dilation_h, dilation_w = _pair(dilation, "dilation")
    return _C.conv2d(
        input,
        weight,
        bias,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        groups,
    )


def conv_transpose2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: PairLike = 1,
    padding: PairLike = 0,
    output_padding: PairLike = 0,
    dilation: PairLike = 1,
    groups: int = 1,
) -> Tensor:
    """Apply a 2D NCHW transposed convolution with CPU kernels and autograd support."""

    stride_h, stride_w = _pair(stride, "stride")
    padding_h, padding_w = _pair(padding, "padding")
    output_padding_h, output_padding_w = _pair(output_padding, "output_padding")
    dilation_h, dilation_w = _pair(dilation, "dilation")
    return _C.conv_transpose2d(
        input,
        weight,
        bias,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        output_padding_h,
        output_padding_w,
        dilation_h,
        dilation_w,
        groups,
    )


def embedding(indices: Tensor, weight: Tensor) -> Tensor:
    """Lookup embedding rows with a native CPU kernel and weight gradients."""

    return _C.embedding(indices, weight)


def max_pool2d(
    input: Tensor,
    kernel_size: PairLike,
    stride: PairLike | None = None,
    padding: PairLike = 0,
    dilation: PairLike = 1,
) -> Tensor:
    """Apply 2D NCHW max pooling with CPU kernels and autograd support."""

    kernel_h, kernel_w = _pair(kernel_size, "kernel_size")
    stride_h, stride_w = _pair(kernel_size if stride is None else stride, "stride")
    padding_h, padding_w = _pair(padding, "padding")
    dilation_h, dilation_w = _pair(dilation, "dilation")
    return _C.max_pool2d(
        input,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
    )


def avg_pool2d(
    input: Tensor,
    kernel_size: PairLike,
    stride: PairLike | None = None,
    padding: PairLike = 0,
    count_include_pad: bool = False,
) -> Tensor:
    """Apply 2D NCHW average pooling with CPU kernels and autograd support."""

    kernel_h, kernel_w = _pair(kernel_size, "kernel_size")
    stride_h, stride_w = _pair(kernel_size if stride is None else stride, "stride")
    padding_h, padding_w = _pair(padding, "padding")
    return _C.avg_pool2d(
        input,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        count_include_pad,
    )


def sum(input: Tensor, axis: AxisLike = None, keepdims: bool = False) -> Tensor:  # noqa: A001
    return _C.sum(input, axis, keepdims)


def mean(input: Tensor, axis: AxisLike = None, keepdims: bool = False) -> Tensor:
    return _C.mean(input, axis, keepdims)


def var(
    input: Tensor,
    axis: int | None = None,
    keepdims: bool = False,
    correction: int = 0,
) -> Tensor:
    return _C.var(input, axis, keepdims, correction)


def variance(
    input: Tensor,
    axis: int | None = None,
    keepdims: bool = False,
    correction: int = 0,
) -> Tensor:
    return _C.variance(input, axis, keepdims, correction)


def std(
    input: Tensor,
    axis: int | None = None,
    keepdims: bool = False,
    correction: int = 0,
) -> Tensor:
    return _C.std(input, axis, keepdims, correction)


def max(input: Tensor, axis: AxisLike = None, keepdims: bool = False) -> Tensor:  # noqa: A001
    return _C.max(input, axis, keepdims)


def min(input: Tensor, axis: AxisLike = None, keepdims: bool = False) -> Tensor:  # noqa: A001
    return _C.min(input, axis, keepdims)


def argmax(input: Tensor, axis: int | None = None, keepdims: bool = False) -> Tensor:
    return _C.argmax(input, axis, keepdims)


def argmin(input: Tensor, axis: int | None = None, keepdims: bool = False) -> Tensor:
    return _C.argmin(input, axis, keepdims)


def all(input: Tensor, axis: AxisLike = None, keepdims: bool = False) -> Tensor:  # noqa: A001
    return _C.all(input, axis, keepdims)


def any(input: Tensor, axis: AxisLike = None, keepdims: bool = False) -> Tensor:  # noqa: A001
    return _C.any(input, axis, keepdims)


def relu(input: Tensor) -> Tensor:
    return _C.relu(input)


def sigmoid(input: Tensor) -> Tensor:
    return _C.sigmoid(input)


def tanh(input: Tensor) -> Tensor:
    return _C.tanh(input)


def exp(input: Tensor) -> Tensor:
    return _C.exp(input)


def log(input: Tensor) -> Tensor:
    return _C.log(input)


def logsumexp(input: Tensor, axis: AxisLike = None, keepdims: bool = False) -> Tensor:
    return _C.logsumexp(input, axis, keepdims)


def softmax(input: Tensor, axis: int = -1) -> Tensor:
    return _C.softmax(input, axis)


def log_softmax(input: Tensor, axis: int = -1) -> Tensor:
    return _C.log_softmax(input, axis)


def log1p(input: Tensor) -> Tensor:
    return _C.log1p(input)


def sqrt(input: Tensor) -> Tensor:
    return _C.sqrt(input)


def rsqrt(input: Tensor) -> Tensor:
    return _C.rsqrt(input)


def sin(input: Tensor) -> Tensor:
    return _C.sin(input)


def cos(input: Tensor) -> Tensor:
    return _C.cos(input)


def tan(input: Tensor) -> Tensor:
    return _C.tan(input)


def asin(input: Tensor) -> Tensor:
    return _C.asin(input)


def acos(input: Tensor) -> Tensor:
    return _C.acos(input)


def atan(input: Tensor) -> Tensor:
    return _C.atan(input)


def abs(input: Tensor) -> Tensor:  # noqa: A001
    return _C.abs(input)


def clamp(input: Tensor, min_value: float, max_value: float) -> Tensor:
    return _C.clamp(input, min_value, max_value)


def clip(input: Tensor, min_value: float, max_value: float) -> Tensor:
    return _C.clamp(input, min_value, max_value)


def astype(input: Tensor, dtype: str) -> Tensor:
    return _C.astype(input, dtype)


def concat(tensors: list[Tensor] | tuple[Tensor, ...], axis: int = 0) -> Tensor:
    return _C.concat(list(tensors), axis)


def stack(tensors: list[Tensor] | tuple[Tensor, ...], axis: int = 0) -> Tensor:
    return _C.stack(list(tensors), axis)


def reshape(input: Tensor, shape: int | tuple[int, ...] | list[int]) -> Tensor:
    return _C.reshape(input, shape)


def flatten(input: Tensor) -> Tensor:
    return _C.flatten(input)


def transpose(input: Tensor, axis0: int | None = None, axis1: int | None = None) -> Tensor:
    return _C.transpose(input, axis0, axis1)


def permute(input: Tensor, axes: tuple[int, ...] | list[int]) -> Tensor:
    return _C.permute(input, axes)


def squeeze(input: Tensor, axis: int | None = None) -> Tensor:
    return _C.squeeze(input, axis)


def unsqueeze(input: Tensor, axis: int) -> Tensor:
    return _C.unsqueeze(input, axis)


__all__ = [
    "add",
    "abs",
    "acos",
    "argmax",
    "argmin",
    "asin",
    "astype",
    "atan",
    "avg_pool2d",
    "all",
    "any",
    "bmm",
    "clamp",
    "clip",
    "concat",
    "cos",
    "conv2d",
    "conv_transpose2d",
    "div",
    "embedding",
    "equal",
    "exp",
    "flatten",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "log",
    "log_softmax",
    "log1p",
    "logsumexp",
    "matmul",
    "max",
    "max_pool2d",
    "maximum",
    "mean",
    "min",
    "minimum",
    "mul",
    "neg",
    "not_equal",
    "pow",
    "permute",
    "relu",
    "reshape",
    "rsqrt",
    "sigmoid",
    "sin",
    "softmax",
    "stack",
    "sub",
    "sum",
    "squeeze",
    "sqrt",
    "std",
    "tan",
    "tanh",
    "transpose",
    "unsqueeze",
    "var",
    "variance",
    "where",
]
