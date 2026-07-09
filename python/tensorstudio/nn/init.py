"""Parameter initialization helpers."""

from __future__ import annotations

import math

from tensorstudio.grad_mode import no_grad
from tensorstudio.tensor import Tensor, normal, ones, uniform, zeros


def _calculate_fan_in_and_fan_out(tensor: Tensor) -> tuple[int, int]:
    if tensor.ndim < 2:
        raise ValueError("fan_in and fan_out require a tensor with at least 2 dimensions")
    if tensor.ndim == 2:
        return tensor.shape[1], tensor.shape[0]
    receptive_field = math.prod(tensor.shape[2:])
    fan_in = tensor.shape[1] * receptive_field
    fan_out = tensor.shape[0] * receptive_field
    return fan_in, fan_out


def _assign(tensor: Tensor, value: Tensor) -> Tensor:
    with no_grad():
        tensor._assign(value)
    return tensor


def zeros_(tensor: Tensor) -> Tensor:
    return _assign(tensor, zeros(tensor.shape, dtype=tensor.dtype))


def ones_(tensor: Tensor) -> Tensor:
    return _assign(tensor, ones(tensor.shape, dtype=tensor.dtype))


def uniform_(tensor: Tensor, a: float = 0.0, b: float = 1.0, seed: int | None = None) -> Tensor:
    return _assign(tensor, uniform(tensor.shape, low=a, high=b, dtype=tensor.dtype, seed=seed))


def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0, seed: int | None = None) -> Tensor:
    if std < 0:
        raise ValueError("std must be non-negative")
    return _assign(
        tensor,
        normal(tensor.shape, mean=mean, stddev=std, dtype=tensor.dtype, seed=seed),
    )


def xavier_uniform_(tensor: Tensor, gain: float = 1.0, seed: int | None = None) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    bound = gain * math.sqrt(6.0 / float(fan_in + fan_out))
    return uniform_(tensor, -bound, bound, seed=seed)


def xavier_normal_(tensor: Tensor, gain: float = 1.0, seed: int | None = None) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    return normal_(tensor, 0.0, std, seed=seed)


def kaiming_uniform_(
    tensor: Tensor,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    seed: int | None = None,
) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    fan = fan_in if mode == "fan_in" else fan_out if mode == "fan_out" else None
    if fan is None:
        raise ValueError("mode must be 'fan_in' or 'fan_out'")
    gain = calculate_gain(nonlinearity, a)
    bound = math.sqrt(3.0) * gain / math.sqrt(float(fan))
    return uniform_(tensor, -bound, bound, seed=seed)


def kaiming_normal_(
    tensor: Tensor,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    seed: int | None = None,
) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    fan = fan_in if mode == "fan_in" else fan_out if mode == "fan_out" else None
    if fan is None:
        raise ValueError("mode must be 'fan_in' or 'fan_out'")
    gain = calculate_gain(nonlinearity, a)
    return normal_(tensor, 0.0, gain / math.sqrt(float(fan)), seed=seed)


def calculate_gain(nonlinearity: str, param: float | None = None) -> float:
    name = nonlinearity.lower()
    if name in {"linear", "sigmoid", "conv1d", "conv2d", "conv_transpose2d"}:
        return 1.0
    if name == "tanh":
        return 5.0 / 3.0
    if name == "relu":
        return math.sqrt(2.0)
    if name == "leaky_relu":
        negative_slope = 0.01 if param is None else param
        return math.sqrt(2.0 / (1.0 + negative_slope**2))
    if name == "selu":
        return 0.75
    raise ValueError(f"unsupported nonlinearity {nonlinearity!r}")


__all__ = [
    "calculate_gain",
    "kaiming_normal_",
    "kaiming_uniform_",
    "normal_",
    "ones_",
    "uniform_",
    "xavier_normal_",
    "xavier_uniform_",
    "zeros_",
]
