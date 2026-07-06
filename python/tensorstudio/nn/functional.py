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


__all__ = ["mse_loss", "relu", "sigmoid", "tanh"]
