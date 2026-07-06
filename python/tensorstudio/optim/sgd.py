"""Stochastic gradient descent optimizer."""

from __future__ import annotations

from collections.abc import Iterable

from tensorstudio.tensor import Tensor


class SGD:
    """Basic SGD optimizer."""

    def __init__(self, params: Iterable[Tensor], lr: float = 0.01) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        self.params = list(params)
        self.lr = lr

    def zero_grad(self) -> None:
        for parameter in self.params:
            parameter.zero_grad()

    def step(self) -> None:
        for parameter in self.params:
            grad = parameter.grad
            if grad is None:
                continue
            parameter._assign(parameter - grad * self.lr)


__all__ = ["SGD"]
