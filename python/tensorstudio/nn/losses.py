"""Loss modules."""

from __future__ import annotations

from tensorstudio.tensor import Tensor

from . import functional as F
from .modules import Module


class MSELoss(Module):
    """Mean squared error loss."""

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input, target)


class L1Loss(Module):
    """Mean absolute error loss."""

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(input, target)


class BCELoss(Module):
    """Binary cross entropy over probabilities."""

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy(input, target, eps=self.eps)


__all__ = ["BCELoss", "L1Loss", "MSELoss"]
