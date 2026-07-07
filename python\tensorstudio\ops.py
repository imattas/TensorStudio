"""Functional tensor operations."""

from __future__ import annotations

from typing import Any

from . import _C
from .tensor import Tensor


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


def sum(input: Tensor) -> Tensor:  # noqa: A001
    return _C.sum(input)


def mean(input: Tensor) -> Tensor:
    return _C.mean(input)


def max(input: Tensor) -> Tensor:  # noqa: A001
    return _C.max(input)


def min(input: Tensor) -> Tensor:  # noqa: A001
    return _C.min(input)


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


def sqrt(input: Tensor) -> Tensor:
    return _C.sqrt(input)


def abs(input: Tensor) -> Tensor:  # noqa: A001
    return _C.abs(input)


def clamp(input: Tensor, min_value: float, max_value: float) -> Tensor:
    return _C.clamp(input, min_value, max_value)


def clip(input: Tensor, min_value: float, max_value: float) -> Tensor:
    return _C.clamp(input, min_value, max_value)


def reshape(input: Tensor, shape: int | tuple[int, ...] | list[int]) -> Tensor:
    return _C.reshape(input, shape)


def flatten(input: Tensor) -> Tensor:
    return _C.flatten(input)


def transpose(input: Tensor) -> Tensor:
    return _C.transpose(input)


__all__ = [
    "add",
    "abs",
    "clamp",
    "clip",
    "div",
    "equal",
    "exp",
    "flatten",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "log",
    "matmul",
    "max",
    "mean",
    "min",
    "mul",
    "neg",
    "not_equal",
    "pow",
    "relu",
    "reshape",
    "sigmoid",
    "sub",
    "sum",
    "sqrt",
    "tanh",
    "transpose",
]
