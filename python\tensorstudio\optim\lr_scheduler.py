"""Learning-rate schedulers."""

from __future__ import annotations

from typing import Protocol, cast


class _OptimizerWithLR(Protocol):
    lr: float


class StepLR:
    """Decay an optimizer's learning rate every ``step_size`` calls."""

    def __init__(self, optimizer: _OptimizerWithLR, step_size: int, gamma: float = 0.1) -> None:
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = 0

    def step(self) -> None:
        self.last_epoch += 1
        if self.last_epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma

    def state_dict(self) -> dict[str, object]:
        return {
            "step_size": self.step_size,
            "gamma": self.gamma,
            "last_epoch": self.last_epoch,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.step_size = cast(int, state["step_size"])
        self.gamma = cast(float, state["gamma"])
        self.last_epoch = cast(int, state["last_epoch"])


class ExponentialLR:
    """Multiply an optimizer's learning rate by ``gamma`` every call."""

    def __init__(self, optimizer: _OptimizerWithLR, gamma: float) -> None:
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        self.optimizer = optimizer
        self.gamma = gamma
        self.last_epoch = 0

    def step(self) -> None:
        self.last_epoch += 1
        self.optimizer.lr *= self.gamma

    def state_dict(self) -> dict[str, object]:
        return {"gamma": self.gamma, "last_epoch": self.last_epoch}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.gamma = cast(float, state["gamma"])
        self.last_epoch = cast(int, state["last_epoch"])


__all__ = ["ExponentialLR", "StepLR"]
