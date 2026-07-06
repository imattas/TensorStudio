"""Optimizer utilities."""

from __future__ import annotations

import math
from collections.abc import Iterable

from tensorstudio.tensor import Tensor


def _parameters(params: Iterable[Tensor]) -> list[Tensor]:
    return list(params)


def clip_grad_value_(params: Iterable[Tensor], clip_value: float) -> None:
    """Clamp gradients in-place to ``[-clip_value, clip_value]``."""

    if clip_value < 0:
        raise ValueError("clip_value must be non-negative")
    for parameter in _parameters(params):
        grad = parameter.grad
        if grad is not None:
            grad._assign(grad.clamp(-clip_value, clip_value))


def clip_grad_norm_(params: Iterable[Tensor], max_norm: float, eps: float = 1e-6) -> float:
    """Scale gradients in-place so the global L2 norm is at most ``max_norm``."""

    if max_norm < 0:
        raise ValueError("max_norm must be non-negative")
    if eps <= 0:
        raise ValueError("eps must be positive")
    gradients = [parameter.grad for parameter in _parameters(params) if parameter.grad is not None]
    total_sq = 0.0
    for grad in gradients:
        assert grad is not None
        values = grad.numpy()
        total_sq += float((values * values).sum())
    total_norm = math.sqrt(total_sq)
    if total_norm > max_norm:
        scale = max_norm / (total_norm + eps)
        for grad in gradients:
            assert grad is not None
            grad._assign(grad * scale)
    return total_norm


__all__ = ["clip_grad_norm_", "clip_grad_value_"]
