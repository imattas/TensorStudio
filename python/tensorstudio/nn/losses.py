"""Loss modules."""

from __future__ import annotations

from tensorstudio.tensor import Tensor

from . import functional as F
from .modules import Module


class MSELoss(Module):
    """Mean squared error loss."""

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input, target)


__all__ = ["MSELoss"]
