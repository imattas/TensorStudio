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


class BCEWithLogitsLoss(Module):
    """Binary cross entropy over logits."""

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(input, target, eps=self.eps)


class CrossEntropyLoss(Module):
    """Multiclass cross entropy over unnormalized logits."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(input, target, reduction=self.reduction)


class HuberLoss(Module):
    """Huber loss with configurable transition point."""

    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        if delta <= 0:
            raise ValueError("delta must be positive")
        self.delta = delta

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.huber_loss(input, target, delta=self.delta)


__all__ = ["BCELoss", "BCEWithLogitsLoss", "CrossEntropyLoss", "HuberLoss", "L1Loss", "MSELoss"]
