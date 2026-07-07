"""Autograd helpers."""

from __future__ import annotations

from . import _C
from .tensor import Tensor


def backward(output: Tensor, gradient: Tensor | None = None, retain_graph: bool = False) -> None:
    """Run reverse-mode automatic differentiation from an output tensor."""

    _C.backward(output, gradient, retain_graph)


__all__ = ["backward"]
