"""Functional neural network operations."""

from __future__ import annotations

import math

from tensorstudio.math import variance
from tensorstudio.ops import avg_pool2d as _avg_pool2d
from tensorstudio.ops import concat
from tensorstudio.ops import conv2d as _conv2d
from tensorstudio.ops import conv_transpose2d as _conv_transpose2d
from tensorstudio.ops import embedding as _embedding
from tensorstudio.ops import log_softmax as _log_softmax
from tensorstudio.ops import max_pool2d as _max_pool2d
from tensorstudio.ops import softmax as _softmax
from tensorstudio.tensor import Tensor, tensor

PairLike = int | tuple[int, int] | list[int]


def _pair(value: PairLike, name: str) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    if len(value) != 2:
        raise ValueError(f"{name} must be an int or a pair of ints")
    return (int(value[0]), int(value[1]))


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
    groups: int = 1,
) -> Tensor:
    return _conv2d(
        input,
        weight,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def conv1d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> Tensor:
    if input.ndim != 3 or weight.ndim != 3:
        raise ValueError("conv1d expects input shape (N, C, L) and weight shape (O, C/groups, K)")
    output = _conv2d(
        input.unsqueeze(-1),
        weight.unsqueeze(-1),
        bias,
        stride=(stride, 1),
        padding=(padding, 0),
        dilation=(dilation, 1),
        groups=groups,
    )
    return output.squeeze(-1)


def conv_transpose2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, int] | list[int] = 1,
    padding: int | tuple[int, int] | list[int] = 0,
    output_padding: int | tuple[int, int] | list[int] = 0,
    dilation: int | tuple[int, int] | list[int] = 1,
    groups: int = 1,
) -> Tensor:
    return _conv_transpose2d(
        input,
        weight,
        bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
    )


def embedding(input: Tensor, weight: Tensor) -> Tensor:
    return _embedding(input, weight)


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


def adaptive_avg_pool2d(input: Tensor, output_size: PairLike) -> Tensor:
    if input.ndim != 4:
        raise ValueError("adaptive_avg_pool2d expects input with shape (N, C, H, W)")
    out_h, out_w = _pair(output_size, "output_size")
    rows: list[Tensor] = []
    for oh in range(out_h):
        h_start = math.floor(oh * input.shape[2] / out_h)
        h_end = math.ceil((oh + 1) * input.shape[2] / out_h)
        cols: list[Tensor] = []
        for ow in range(out_w):
            w_start = math.floor(ow * input.shape[3] / out_w)
            w_end = math.ceil((ow + 1) * input.shape[3] / out_w)
            cols.append(input[:, :, h_start:h_end, w_start:w_end].mean(axis=(2, 3), keepdims=True))
        rows.append(concat(cols, axis=3))
    return concat(rows, axis=2)


def adaptive_max_pool2d(input: Tensor, output_size: PairLike) -> Tensor:
    if input.ndim != 4:
        raise ValueError("adaptive_max_pool2d expects input with shape (N, C, H, W)")
    out_h, out_w = _pair(output_size, "output_size")
    rows: list[Tensor] = []
    for oh in range(out_h):
        h_start = math.floor(oh * input.shape[2] / out_h)
        h_end = math.ceil((oh + 1) * input.shape[2] / out_h)
        cols: list[Tensor] = []
        for ow in range(out_w):
            w_start = math.floor(ow * input.shape[3] / out_w)
            w_end = math.ceil((ow + 1) * input.shape[3] / out_w)
            cols.append(input[:, :, h_start:h_end, w_start:w_end].max(axis=(2, 3), keepdims=True))
        rows.append(concat(cols, axis=3))
    return concat(rows, axis=2)


def global_avg_pool2d(input: Tensor, keepdims: bool = False) -> Tensor:
    if input.ndim != 4:
        raise ValueError("global_avg_pool2d expects input with shape (N, C, H, W)")
    return input.mean(axis=(2, 3), keepdims=keepdims)


def global_max_pool2d(input: Tensor, keepdims: bool = False) -> Tensor:
    if input.ndim != 4:
        raise ValueError("global_max_pool2d expects input with shape (N, C, H, W)")
    return input.max(axis=(2, 3), keepdims=keepdims)


def batch_norm(
    input: Tensor,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    running_mean: Tensor | None = None,
    running_var: Tensor | None = None,
    training: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    if input.ndim not in {2, 3, 4}:
        raise ValueError("batch_norm expects input with shape (N, C), (N, C, L), or (N, C, H, W)")
    axes = (0,) if input.ndim == 2 else (0, *range(2, input.ndim))
    broadcast_shape = (1, input.shape[1], *([1] * (input.ndim - 2)))
    if training or running_mean is None or running_var is None:
        mean = input.mean(axis=axes, keepdims=False)
        var = variance(input, axis=axes, keepdims=False)
        if running_mean is not None and running_var is not None:
            updated_mean = running_mean * (1.0 - momentum) + mean.detach() * momentum
            updated_var = running_var * (1.0 - momentum) + var.detach() * momentum
            running_mean._assign(updated_mean)
            running_var._assign(updated_var)
    else:
        mean = running_mean
        var = running_var
    output = (input - mean.reshape(broadcast_shape)) / (var.reshape(broadcast_shape) + eps).sqrt()
    if weight is not None:
        output = output * weight.reshape(broadcast_shape)
    if bias is not None:
        output = output + bias.reshape(broadcast_shape)
    return output


def layer_norm(
    input: Tensor,
    normalized_shape: int | tuple[int, ...] | list[int],
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor:
    shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    if tuple(input.shape[-len(shape) :]) != shape:
        raise ValueError("layer_norm normalized_shape must match trailing input dimensions")
    axes = tuple(range(input.ndim - len(shape), input.ndim))
    mean = input.mean(axis=axes, keepdims=True)
    var = variance(input, axis=axes, keepdims=True)
    output = (input - mean) / (var + eps).sqrt()
    if weight is not None:
        output = output * weight
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


def gelu(input: Tensor) -> Tensor:
    coeff = math.sqrt(2.0 / math.pi)
    return 0.5 * input * (1.0 + (coeff * (input + 0.044715 * (input**3))).tanh())


def elu(input: Tensor, alpha: float = 1.0) -> Tensor:
    return (input > 0.0).where(input, alpha * (input.exp() - 1.0))


def selu(input: Tensor) -> Tensor:
    return 1.0507009873554805 * elu(input, alpha=1.6732632423543772)


def silu(input: Tensor) -> Tensor:
    return input * input.sigmoid()


def mish(input: Tensor) -> Tensor:
    return input * softplus(input).tanh()


def softmax(input: Tensor, axis: int = -1) -> Tensor:
    return _softmax(input, axis=axis)


def log_softmax(input: Tensor, axis: int = -1) -> Tensor:
    return _log_softmax(input, axis=axis)


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


def _apply_reduction(loss: Tensor, reduction: str, batch_size: int | None = None) -> Tensor:
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "batchmean":
        if batch_size is None:
            raise ValueError("batchmean reduction requires a batch size")
        return loss.sum() / float(batch_size)
    if reduction == "none":
        return loss
    raise ValueError("reduction must be 'mean', 'sum', 'batchmean', or 'none'")


def cross_entropy(
    input: Tensor,
    target: Tensor,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> Tensor:
    if input.ndim != 2:
        raise ValueError("cross_entropy input must have shape (batch, classes)")
    if target.shape != (input.shape[0],):
        raise ValueError("cross_entropy target must have shape (batch,)")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("cross_entropy reduction must be 'mean', 'sum', or 'none'")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must satisfy 0 <= label_smoothing < 1")

    one_hot = _one_hot_from_targets(target, input.shape[1], input.dtype)
    if label_smoothing:
        one_hot = one_hot * (1.0 - label_smoothing) + label_smoothing / float(input.shape[1])
    per_sample = -(one_hot * log_softmax(input, axis=1)).sum(axis=1)
    return _apply_reduction(per_sample, reduction)


def nll_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    if input.ndim != 2:
        raise ValueError("nll_loss input must have shape (batch, classes)")
    if target.shape != (input.shape[0],):
        raise ValueError("nll_loss target must have shape (batch,)")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("nll_loss reduction must be 'mean', 'sum', or 'none'")
    one_hot = _one_hot_from_targets(target, input.shape[1], input.dtype)
    per_sample = -(one_hot * input).sum(axis=1)
    return _apply_reduction(per_sample, reduction)


def kl_div(
    input: Tensor,
    target: Tensor,
    reduction: str = "mean",
    log_target: bool = False,
) -> Tensor:
    if input.shape != target.shape:
        raise ValueError("kl_div input and target must have the same shape")
    if reduction not in {"mean", "sum", "batchmean", "none"}:
        raise ValueError("kl_div reduction must be 'mean', 'sum', 'batchmean', or 'none'")
    if log_target:
        loss = target.exp() * (target - input)
    else:
        safe_target = target.clamp(1e-12, 1.0)
        loss = safe_target * (safe_target.log() - input)
    return _apply_reduction(loss, reduction, batch_size=input.shape[0] if input.ndim else 1)


def focal_loss(
    input: Tensor,
    target: Tensor,
    gamma: float = 2.0,
    alpha: float | None = None,
    reduction: str = "mean",
) -> Tensor:
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    if input.ndim != 2:
        raise ValueError("focal_loss input must have shape (batch, classes)")
    if target.shape != (input.shape[0],):
        raise ValueError("focal_loss target must have shape (batch,)")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("focal_loss reduction must be 'mean', 'sum', or 'none'")
    one_hot = _one_hot_from_targets(target, input.shape[1], input.dtype)
    probs = softmax(input, axis=1)
    log_probs = log_softmax(input, axis=1)
    pt = (one_hot * probs).sum(axis=1).clamp(1e-12, 1.0)
    ce = -(one_hot * log_probs).sum(axis=1)
    loss = ((1.0 - pt) ** gamma) * ce
    if alpha is not None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        loss = loss * alpha
    return _apply_reduction(loss, reduction)


def cosine_embedding_loss(
    input1: Tensor,
    input2: Tensor,
    target: Tensor,
    margin: float = 0.0,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> Tensor:
    if input1.shape != input2.shape:
        raise ValueError("cosine_embedding_loss input tensors must have the same shape")
    if input1.ndim == 1:
        left = input1.reshape((1, input1.shape[0]))
        right = input2.reshape((1, input2.shape[0]))
    elif input1.ndim == 2:
        left = input1
        right = input2
    else:
        raise ValueError("cosine_embedding_loss expects 1D or 2D inputs")
    if target.shape not in {(left.shape[0],), ()}:
        raise ValueError("cosine_embedding_loss target must be scalar or have shape (batch,)")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("cosine_embedding_loss reduction must be 'mean', 'sum', or 'none'")

    dot = (left * right).sum(axis=1)
    left_norm = (left * left).sum(axis=1).sqrt()
    right_norm = (right * right).sum(axis=1).sqrt()
    cosine = dot / (left_norm * right_norm + eps)
    raw_targets = target.tolist()
    target_values = [raw_targets] if isinstance(raw_targets, (int, float)) else list(raw_targets)
    if len(target_values) == 1 and left.shape[0] != 1:
        target_values = target_values * left.shape[0]
    if len(target_values) != left.shape[0]:
        raise ValueError("cosine_embedding_loss target length must match batch size")
    positive_mask = tensor(
        [1.0 if value >= 0 else 0.0 for value in target_values],
        dtype=input1.dtype,
    )
    negative_mask = 1.0 - positive_mask
    positive = 1.0 - cosine
    negative = (cosine - margin).relu()
    loss = positive_mask * positive + negative_mask * negative
    return _apply_reduction(loss, reduction)


def huber_loss(input: Tensor, target: Tensor, delta: float = 1.0) -> Tensor:
    if delta <= 0:
        raise ValueError("delta must be positive")
    diff = input - target
    abs_diff = diff.abs()
    quadratic = (diff * diff) * 0.5
    linear = delta * (abs_diff - 0.5 * delta)
    return (quadratic * (abs_diff <= delta) + linear * (abs_diff > delta)).mean()


__all__ = [
    "adaptive_avg_pool2d",
    "adaptive_max_pool2d",
    "batch_norm",
    "binary_cross_entropy",
    "binary_cross_entropy_with_logits",
    "avg_pool2d",
    "conv1d",
    "conv2d",
    "conv_transpose2d",
    "cosine_embedding_loss",
    "cross_entropy",
    "elu",
    "embedding",
    "focal_loss",
    "gelu",
    "global_avg_pool2d",
    "global_max_pool2d",
    "huber_loss",
    "kl_div",
    "l1_loss",
    "layer_norm",
    "leaky_relu",
    "linear",
    "log_softmax",
    "max_pool2d",
    "mish",
    "mse_loss",
    "nll_loss",
    "relu",
    "selu",
    "sigmoid",
    "silu",
    "softmax",
    "softplus",
    "tanh",
]
