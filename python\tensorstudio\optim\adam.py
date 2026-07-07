"""Adam optimizer."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

from tensorstudio.tensor import Tensor, zeros


class Adam:
    """Adam optimizer with bias correction and optional coupled weight decay."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        beta1, beta2 = betas
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("betas must be in [0, 1)")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self._m = [zeros(parameter.shape, dtype=parameter.dtype) for parameter in self.params]
        self._v = [zeros(parameter.shape, dtype=parameter.dtype) for parameter in self.params]

    def zero_grad(self) -> None:
        for parameter in self.params:
            parameter.zero_grad()

    def step(self) -> None:
        self.t += 1
        for index, parameter in enumerate(self.params):
            grad = parameter.grad
            if grad is None:
                continue
            if self.weight_decay:
                grad = grad + parameter * self.weight_decay
            self._m[index] = self._m[index] * self.beta1 + grad * (1.0 - self.beta1)
            self._v[index] = self._v[index] * self.beta2 + (grad * grad) * (1.0 - self.beta2)
            m_hat = self._m[index] / (1.0 - self.beta1**self.t)
            v_hat = self._v[index] / (1.0 - self.beta2**self.t)
            parameter._assign(parameter - self.lr * m_hat / ((v_hat**0.5) + self.eps))

    def state_dict(self) -> dict[str, object]:
        return {
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "t": self.t,
            "m": [value.detach().clone() for value in self._m],
            "v": [value.detach().clone() for value in self._v],
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.lr = cast(float, state["lr"])
        self.beta1 = cast(float, state["beta1"])
        self.beta2 = cast(float, state["beta2"])
        self.eps = cast(float, state["eps"])
        self.weight_decay = cast(float, state.get("weight_decay", 0.0))
        self.t = cast(int, state["t"])
        m_state = cast(list[Tensor], state["m"])
        v_state = cast(list[Tensor], state["v"])
        self._m = list(m_state)
        self._v = list(v_state)


class AdamW(Adam):
    """AdamW optimizer with decoupled weight decay."""

    def step(self) -> None:
        self.t += 1
        for index, parameter in enumerate(self.params):
            grad = parameter.grad
            if grad is None:
                continue
            self._m[index] = self._m[index] * self.beta1 + grad * (1.0 - self.beta1)
            self._v[index] = self._v[index] * self.beta2 + (grad * grad) * (1.0 - self.beta2)
            m_hat = self._m[index] / (1.0 - self.beta1**self.t)
            v_hat = self._v[index] / (1.0 - self.beta2**self.t)
            update = m_hat / ((v_hat**0.5) + self.eps)
            if self.weight_decay:
                update = update + parameter * self.weight_decay
            parameter._assign(parameter - self.lr * update)


__all__ = ["Adam", "AdamW"]
