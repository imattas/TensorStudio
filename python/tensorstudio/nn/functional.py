"""Functional neural network operations."""

from __future__ import annotations

from tensorstudio.tensor import Tensor


def relu(input: Tensor) -> Tensor:
    return input.relu()


def sigmoid(input: Tensor) -> Tensor:
    return input.sigmoid()


def tanh(input: Tensor) -> Tensor:
    return input.tanh()


def mse_loss(input: Tensor, target: Tensor) -> Tensor:
    diff = input - target
    return (diff * diff).mean()


def l1_loss(input: Tensor, target: Tensor) -> Tensor:
    return (input - target).abs().mean()


def binary_cross_entropy(input: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    clipped = input.clamp(eps, 1.0 - eps)
    return -(target * clipped.log() + (1.0 - target) * (1.0 - clipped).log()).mean()


__all__ = ["binary_cross_entropy", "l1_loss", "mse_loss", "relu", "sigmoid", "tanh"]
