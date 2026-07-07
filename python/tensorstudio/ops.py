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


def neg(input: Tensor) -> Tensor:
    return _C.neg(input)


def pow(input: Tensor, exponent: float) -> Tensor:
    return _C.pow(input, exponent)


def matmul(left: Tensor, right: Any) -> Tensor:
    return _C.matmul(left, right)


def conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: PairLike = 1,
    padding: PairLike = 0,
    dilation: PairLike = 1,
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
    )


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


def max(input: Tensor, axis: AxisLike = None, keepdims: bool = False) -> Tensor:  # noqa: A001
    return _C.max(input, axis, keepdims)


def min(input: Tensor, axis: AxisLike = None, keepdims: bool = False) -> Tensor:  # noqa: A001
    return _C.min(input, axis, keepdims)


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


def transpose(input: Tensor) -> Tensor:
    return _C.transpose(input)


__all__ = [
    "add",
    "abs",
    "acos",
    "asin",
    "astype",
    "atan",
    "avg_pool2d",
    "clamp",
    "clip",
    "concat",
    "cos",
    "conv2d",
    "div",
    "equal",
    "exp",
    "flatten",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "log",
    "log1p",
    "matmul",
    "max",
    "max_pool2d",
    "mean",
    "min",
    "mul",
    "neg",
    "not_equal",
    "pow",
    "relu",
    "reshape",
    "rsqrt",
    "sigmoid",
    "sin",
    "stack",
    "sub",
    "sum",
    "sqrt",
    "tan",
    "tanh",
    "transpose",
]
