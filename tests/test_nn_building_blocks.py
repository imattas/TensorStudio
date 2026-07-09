from __future__ import annotations

import math

import numpy as np
import pytest
import tensorstudio as ts
from tensorstudio import nn


def _grouped_conv2d_reference(
    x: np.ndarray,
    w: np.ndarray,
    bias: np.ndarray | None = None,
    groups: int = 1,
    padding: int = 0,
) -> np.ndarray:
    padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    batch, in_channels, height, width = x.shape
    out_channels, in_per_group, kernel_h, kernel_w = w.shape
    out = np.zeros((batch, out_channels, height, width), dtype=np.float64)
    out_per_group = out_channels // groups
    for n in range(batch):
        for oc in range(out_channels):
            group = oc // out_per_group
            for oh in range(height):
                for ow in range(width):
                    total = 0.0 if bias is None else float(bias[oc])
                    for local_ic in range(in_per_group):
                        ic = group * in_per_group + local_ic
                        window = padded[n, ic, oh : oh + kernel_h, ow : ow + kernel_w]
                        total += float(np.sum(window * w[oc, local_ic]))
                    out[n, oc, oh, ow] = total
    return out


def _conv_transpose2d_reference(
    x: np.ndarray,
    w: np.ndarray,
    bias: np.ndarray | None = None,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
) -> np.ndarray:
    batch, in_channels, in_h, in_w = x.shape
    _, out_channels, kernel_h, kernel_w = w.shape
    out_h = (in_h - 1) * stride - 2 * padding + kernel_h + output_padding
    out_w = (in_w - 1) * stride - 2 * padding + kernel_w + output_padding
    out = np.zeros((batch, out_channels, out_h, out_w), dtype=np.float64)
    for n in range(batch):
        for ic in range(in_channels):
            for ih in range(in_h):
                for iw in range(in_w):
                    for oc in range(out_channels):
                        for kh in range(kernel_h):
                            oh = ih * stride - padding + kh
                            if oh < 0 or oh >= out_h:
                                continue
                            for kw in range(kernel_w):
                                ow = iw * stride - padding + kw
                                if 0 <= ow < out_w:
                                    out[n, oc, oh, ow] += x[n, ic, ih, iw] * w[ic, oc, kh, kw]
    if bias is not None:
        out += bias.reshape(1, out_channels, 1, 1)
    return out


def test_grouped_depthwise_conv_and_conv1d() -> None:
    x_np = np.arange(1, 1 + 4 * 3 * 3, dtype=np.float64).reshape(1, 4, 3, 3)
    w_np = np.ones((4, 2, 3, 3), dtype=np.float64)
    bias_np = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    x = ts.tensor(x_np.tolist(), dtype="float64", requires_grad=True)
    w = ts.tensor(w_np.tolist(), dtype="float64", requires_grad=True)
    b = ts.tensor(bias_np.tolist(), dtype="float64", requires_grad=True)

    y = ts.conv2d(x, w, b, padding=1, groups=2)

    np.testing.assert_allclose(y.numpy(), _grouped_conv2d_reference(x_np, w_np, bias_np, 2, 1))
    y.sum().backward()
    assert x.grad.shape == x.shape
    assert w.grad.shape == w.shape
    assert b.grad.tolist() == pytest.approx([9.0, 9.0, 9.0, 9.0])

    depthwise = nn.DepthwiseConv2d(4, kernel_size=3, padding=1)
    assert depthwise.groups == 4
    assert depthwise(x.detach()).shape == (1, 4, 3, 3)

    conv1d = nn.Conv1d(2, 3, kernel_size=3, padding=1)
    assert conv1d(ts.ones((2, 2, 5))).shape == (2, 3, 5)


def test_conv_transpose2d_matches_reference_and_backpropagates() -> None:
    x_np = np.arange(1, 1 + 2 * 2 * 2, dtype=np.float64).reshape(1, 2, 2, 2)
    w_np = np.arange(1, 1 + 2 * 3 * 2 * 2, dtype=np.float64).reshape(2, 3, 2, 2) / 10.0
    bias_np = np.array([0.5, -0.5, 1.0], dtype=np.float64)
    x = ts.tensor(x_np.tolist(), dtype="float64", requires_grad=True)
    w = ts.tensor(w_np.tolist(), dtype="float64", requires_grad=True)
    b = ts.tensor(bias_np.tolist(), dtype="float64", requires_grad=True)

    y = ts.conv_transpose2d(x, w, b, stride=2, padding=1, output_padding=1)

    expected = _conv_transpose2d_reference(
        x_np,
        w_np,
        bias_np,
        stride=2,
        padding=1,
        output_padding=1,
    )
    np.testing.assert_allclose(y.numpy(), expected)
    y.mean().backward()
    assert x.grad.shape == x.shape
    assert w.grad.shape == w.shape
    assert b.grad.shape == b.shape


def test_embedding_accumulates_repeated_index_gradients() -> None:
    layer = nn.Embedding(4, 3)
    layer.weight._assign(ts.ones((4, 3)))
    indices = ts.tensor([[1, 2], [1, 3]], dtype="int64")

    output = layer(indices)
    output.sum().backward()

    assert output.shape == (2, 2, 3)
    np.testing.assert_allclose(
        layer.weight.grad.numpy(),
        np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
    )


def test_normalization_buffers_layernorm_and_state_dict() -> None:
    bn = nn.BatchNorm2d(3)
    x = ts.tensor(np.arange(2 * 3 * 2 * 2, dtype=np.float64).reshape(2, 3, 2, 2), dtype="float64")
    y = bn(x)

    assert y.shape == x.shape
    assert [name for name, _ in bn.named_buffers()] == ["running_mean", "running_var"]
    assert [parameter.shape for parameter in bn.parameters()] == [(3,), (3,)]
    state = bn.state_dict()
    assert {"weight", "bias", "running_mean", "running_var"} <= set(state)
    bn.eval()
    assert bn(x).shape == x.shape

    ln = nn.LayerNorm((2, 2))
    normalized = ln(ts.ones((2, 3, 2, 2)))
    assert normalized.shape == (2, 3, 2, 2)


def test_initializers_activation_modules_and_pooling_summary() -> None:
    weight = ts.zeros((4, 3))
    nn.init.xavier_uniform_(weight, seed=123)
    assert np.max(np.abs(weight.numpy())) <= math.sqrt(6.0 / 7.0) + 1e-6
    nn.init.ones_(weight)
    assert weight.sum().item() == pytest.approx(12.0)

    values = ts.tensor([-1.0, 0.0, 1.0])
    np.testing.assert_allclose(
        nn.SiLU()(values).numpy(),
        values.numpy() / (1.0 + np.exp(-values.numpy())),
        rtol=1e-6,
    )
    assert nn.GELU()(values).shape == values.shape
    assert nn.ELU()(values).shape == values.shape
    assert nn.SELU()(values).shape == values.shape
    assert nn.Mish()(values).shape == values.shape

    image = ts.arange(1, 17).reshape((1, 1, 4, 4))
    assert nn.AdaptiveAvgPool2d((2, 2))(image).shape == (1, 1, 2, 2)
    assert nn.AdaptiveMaxPool2d((2, 2))(image).shape == (1, 1, 2, 2)
    assert nn.GlobalAvgPool2d()(image).shape == (1, 1)
    assert nn.GlobalMaxPool2d(keepdims=True)(image).shape == (1, 1, 1, 1)

    model = nn.Sequential(
        nn.Conv2d(1, 2, 3, padding=1),
        nn.GlobalAvgPool2d(),
        nn.Flatten(),
        nn.Linear(2, 2),
    )
    summary = nn.summary(model, input_shape=(1, 1, 4, 4))
    assert summary["total_parameters"] == model.parameter_count()
    assert summary["estimated_total_bytes"] > 0
    assert summary["rows"][0]["output_shape"] == (1, 2, 4, 4)


def test_expanded_losses() -> None:
    logits = ts.tensor([[1.0, 2.0, 3.0], [1.0, 0.0, -1.0]], dtype="float64", requires_grad=True)
    target = ts.tensor([2, 0], dtype="int64")
    log_probs = logits.log_softmax(axis=1)
    probs = logits.softmax(axis=1).detach()

    ce = nn.CrossEntropyLoss(label_smoothing=0.1)(logits, target)
    nll = nn.NLLLoss()(log_probs, target)
    kl = nn.KLDivLoss(reduction="batchmean")(log_probs, probs)
    focal = nn.FocalLoss(gamma=2.0)(logits, target)
    cosine = nn.CosineEmbeddingLoss()(
        ts.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True),
        ts.tensor([[1.0, 0.0], [-1.0, 0.0]], requires_grad=True),
        ts.tensor([1.0, -1.0]),
    )
    loss = ce + nll + kl + focal + cosine

    assert loss.item() >= 0.0
    loss.backward()
    assert logits.grad.shape == logits.shape
