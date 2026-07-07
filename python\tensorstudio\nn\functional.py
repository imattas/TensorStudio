"""Functional neural network operations."""

from __future__ import annotations

from tensorstudio.tensor import Tensor


def linear(input: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    output = input @ weight.T
    if bias is not None:
        output = output + bias
    return output


def relu(input: Tensor) -> Tensor:
    return input.relu()


def leaky_relu(input: Tensor, negative_slope: float = 0.01) -> Tensor:
    return input.relu() - (-input).relu() * negative_slope


def sigmoid(input: Tensor) -> Tensor:
    return input.sigmoid()


def tanh(input: Tensor) -> Tensor:
    return input.tanh()


def softplus(input: Tensor) -> Tensor:
    return (1.0 + input.exp()).log()


def mse_loss(input: Tensor, target: Tensor) -> Tensor:
    diff = input - target
    return (diff * diff).mean()


def l1_loss(input: Tensor, target: Tensor) -> Tensor:
    return (input - target).abs().mean()


def binary_cross_entropy(input: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    clipped = input.clamp(eps, 1.0 - eps)
    return -(target * clipped.log() + (1.0 - target) * (1.0 - clipped).log()).mean()


def binary_cross_entropy_with_logits(input: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    return binary_cross_entropy(input.sigmoid(), target, eps=eps)


def huber_loss(input: Tensor, target: Tensor, delta: float = 1.0) -> Tensor:
    if delta <= 0:
        raise ValueError("delta must be positive")
    diff = input - target
    abs_diff = diff.abs()
    quadratic = (diff * diff) * 0.5
    linear = delta * (abs_diff - 0.5 * delta)
    return (quadratic * (abs_diff <= delta) + linear * (abs_diff > delta)).mean()


__all__ = [
    "binary_cross_entropy",
    "binary_cross_entropy_with_logits",
    "huber_loss",
    "l1_loss",
    "leaky_relu",
    "linear",
    "mse_loss",
    "relu",
    "sigmoid",
    "softplus",
    "tanh",
]
