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


def reshape(input: Tensor, shape: int | tuple[int, ...] | list[int]) -> Tensor:
    return _C.reshape(input, shape)


def flatten(input: Tensor) -> Tensor:
    return _C.flatten(input)


def transpose(input: Tensor) -> Tensor:
    return _C.transpose(input)


__all__ = [
    "add",
    "div",
    "exp",
    "flatten",
    "log",
    "matmul",
    "mean",
    "mul",
    "neg",
    "pow",
    "relu",
    "reshape",
    "sigmoid",
    "sub",
    "sum",
    "tanh",
    "transpose",
]
