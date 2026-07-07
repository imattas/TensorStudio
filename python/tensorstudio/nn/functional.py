"""Functional neural network operations."""

from __future__ import annotations

from tensorstudio.ops import avg_pool2d as _avg_pool2d
from tensorstudio.ops import conv2d as _conv2d
from tensorstudio.ops import max_pool2d as _max_pool2d
from tensorstudio.tensor import Tensor, tensor


def linear(input: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    output = input @ weight.T
    if bias is not None:
        output = output + bias
    return output


def conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, int] | list[int] = 1,
    padding: int | tuple[int, int] | list[int] = 0,
    dilation: int | tuple[int, int] | list[int] = 1,
) -> Tensor:
    return _conv2d(input, weight, bias, stride=stride, padding=padding, dilation=dilation)


def max_pool2d(
    input: Tensor,
    kernel_size: int | tuple[int, int] | list[int],
    stride: int | tuple[int, int] | list[int] | None = None,
    padding: int | tuple[int, int] | list[int] = 0,
    dilation: int | tuple[int, int] | list[int] = 1,
) -> Tensor:
    return _max_pool2d(input, kernel_size, stride=stride, padding=padding, dilation=dilation)


def avg_pool2d(
    input: Tensor,
    kernel_size: int | tuple[int, int] | list[int],
    stride: int | tuple[int, int] | list[int] | None = None,
    padding: int | tuple[int, int] | list[int] = 0,
    count_include_pad: bool = False,
) -> Tensor:
    return _avg_pool2d(
        input,
        kernel_size,
        stride=stride,
        padding=padding,
        count_include_pad=count_include_pad,
    )


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


def softmax(input: Tensor, axis: int = -1) -> Tensor:
    shifted = input - input.max(axis=axis, keepdims=True).detach()
    exp_values = shifted.exp()
    return exp_values / exp_values.sum(axis=axis, keepdims=True)


def log_softmax(input: Tensor, axis: int = -1) -> Tensor:
    shifted = input - input.max(axis=axis, keepdims=True).detach()
    return shifted - shifted.exp().sum(axis=axis, keepdims=True).log()


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


def _one_hot_from_targets(target: Tensor, num_classes: int, dtype: str) -> Tensor:
    if target.ndim != 1:
        raise ValueError("cross_entropy target must be a 1D tensor of class indices")
    if target.dtype not in {"int32", "int64"}:
        raise ValueError("cross_entropy target dtype must be int32 or int64")
    labels = target.tolist()
    rows: list[list[float]] = []
    for raw_label in labels:
        if isinstance(raw_label, bool) or not isinstance(raw_label, int):
            raise ValueError("cross_entropy target values must be integer class indices")
        if raw_label < 0 or raw_label >= num_classes:
            raise ValueError("cross_entropy target contains a class index out of range")
        row = [0.0] * num_classes
        row[raw_label] = 1.0
        rows.append(row)
    return tensor(rows, dtype=dtype)


def cross_entropy(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    if input.ndim != 2:
        raise ValueError("cross_entropy input must have shape (batch, classes)")
    if target.shape != (input.shape[0],):
        raise ValueError("cross_entropy target must have shape (batch,)")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("cross_entropy reduction must be 'mean', 'sum', or 'none'")

    one_hot = _one_hot_from_targets(target, input.shape[1], input.dtype)
    per_sample = -(one_hot * log_softmax(input, axis=1)).sum(axis=1)
    if reduction == "mean":
        return per_sample.mean()
    if reduction == "sum":
        return per_sample.sum()
    return per_sample


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
    "avg_pool2d",
    "conv2d",
    "cross_entropy",
    "huber_loss",
    "l1_loss",
    "leaky_relu",
    "linear",
    "log_softmax",
    "max_pool2d",
    "mse_loss",
    "relu",
    "sigmoid",
    "softmax",
    "softplus",
    "tanh",
]
