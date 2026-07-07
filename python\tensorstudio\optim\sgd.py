"""Stochastic gradient descent optimizer."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

from tensorstudio.tensor import Tensor


class SGD:
    """Stochastic gradient descent with optional momentum and weight decay."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        if momentum < 0:
            raise ValueError("momentum must be non-negative")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocity: list[Tensor | None] = [None for _ in self.params]

    def zero_grad(self) -> None:
        for parameter in self.params:
            parameter.zero_grad()

    def step(self) -> None:
        for index, parameter in enumerate(self.params):
            grad = parameter.grad
            if grad is None:
                continue
            update = grad + parameter * self.weight_decay if self.weight_decay else grad
            if self.momentum:
                previous = self._velocity[index]
                velocity = update if previous is None else previous * self.momentum + update
                self._velocity[index] = velocity.detach().clone()
                update = velocity
            parameter._assign(parameter - update * self.lr)

    def state_dict(self) -> dict[str, object]:
        return {
            "lr": self.lr,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "velocity": [
                value.detach().clone() if value is not None else None for value in self._velocity
            ],
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.lr = cast(float, state["lr"])
        self.momentum = cast(float, state["momentum"])
        self.weight_decay = cast(float, state["weight_decay"])
        velocity = state.get("velocity", [])
        self._velocity = (
            cast(list[Tensor | None], list(velocity))
            if isinstance(velocity, list)
            else [None for _ in self.params]
        )


__all__ = ["SGD"]
