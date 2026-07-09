from __future__ import annotations

import numpy as np
import pytest
import tensorstudio as ts


def _numpy_conv2d(
    input: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
) -> np.ndarray:
    batch, in_channels, input_h, input_w = input.shape
    out_channels, weight_channels, kernel_h, kernel_w = weight.shape
    assert in_channels == weight_channels
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation
    effective_h = dilation_h * (kernel_h - 1) + 1
    effective_w = dilation_w * (kernel_w - 1) + 1
    out_h = (input_h + 2 * padding_h - effective_h) // stride_h + 1
    out_w = (input_w + 2 * padding_w - effective_w) // stride_w + 1
    output = np.zeros((batch, out_channels, out_h, out_w), dtype=input.dtype)
    for n in range(batch):
        for oc in range(out_channels):
            for oh in range(out_h):
                for ow in range(out_w):
                    acc = 0.0 if bias is None else float(bias[oc])
                    for ic in range(in_channels):
                        for kh in range(kernel_h):
                            ih = oh * stride_h - padding_h + kh * dilation_h
                            if ih < 0 or ih >= input_h:
                                continue
                            for kw in range(kernel_w):
                                iw = ow * stride_w - padding_w + kw * dilation_w
                                if iw < 0 or iw >= input_w:
                                    continue
                                acc += float(input[n, ic, ih, iw] * weight[oc, ic, kh, kw])
                    output[n, oc, oh, ow] = acc
    return output


def _numpy_max_pool2d(
    input: np.ndarray,
    kernel_size: tuple[int, int],
    stride: tuple[int, int] | None = None,
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
) -> np.ndarray:
    stride = kernel_size if stride is None else stride
    batch, channels, input_h, input_w = input.shape
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation
    effective_h = dilation_h * (kernel_h - 1) + 1
    effective_w = dilation_w * (kernel_w - 1) + 1
    out_h = (input_h + 2 * padding_h - effective_h) // stride_h + 1
    out_w = (input_w + 2 * padding_w - effective_w) // stride_w + 1
    output = np.empty((batch, channels, out_h, out_w), dtype=input.dtype)
    for n in range(batch):
        for c in range(channels):
            for oh in range(out_h):
                for ow in range(out_w):
                    values: list[float] = []
                    for kh in range(kernel_h):
                        ih = oh * stride_h - padding_h + kh * dilation_h
                        if ih < 0 or ih >= input_h:
                            continue
                        for kw in range(kernel_w):
                            iw = ow * stride_w - padding_w + kw * dilation_w
                            if 0 <= iw < input_w:
                                values.append(float(input[n, c, ih, iw]))
                    output[n, c, oh, ow] = max(values)
    return output


def _numpy_avg_pool2d(
    input: np.ndarray,
    kernel_size: tuple[int, int],
    stride: tuple[int, int] | None = None,
    padding: tuple[int, int] = (0, 0),
    count_include_pad: bool = False,
) -> np.ndarray:
    stride = kernel_size if stride is None else stride
    batch, channels, input_h, input_w = input.shape
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    out_h = (input_h + 2 * padding_h - kernel_h) // stride_h + 1
    out_w = (input_w + 2 * padding_w - kernel_w) // stride_w + 1
    output = np.empty((batch, channels, out_h, out_w), dtype=input.dtype)
    for n in range(batch):
        for c in range(channels):
            for oh in range(out_h):
                for ow in range(out_w):
                    acc = 0.0
                    valid_count = 0
                    for kh in range(kernel_h):
                        ih = oh * stride_h - padding_h + kh
                        if ih < 0 or ih >= input_h:
                            continue
                        for kw in range(kernel_w):
                            iw = ow * stride_w - padding_w + kw
                            if 0 <= iw < input_w:
                                acc += float(input[n, c, ih, iw])
                                valid_count += 1
                    denom = kernel_h * kernel_w if count_include_pad else valid_count
                    output[n, c, oh, ow] = acc / denom
    return output


def test_elementwise_ops_match_numpy() -> None:
    a = ts.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = ts.tensor([10.0, 20.0])

    np.testing.assert_allclose((a + b).numpy(), np.array([[11.0, 22.0], [13.0, 24.0]]))
    np.testing.assert_allclose((a - 1).numpy(), np.array([[0.0, 1.0], [2.0, 3.0]]))
    np.testing.assert_allclose((a * 2).numpy(), np.array([[2.0, 4.0], [6.0, 8.0]]))
    np.testing.assert_allclose((a / 2).numpy(), np.array([[0.5, 1.0], [1.5, 2.0]]))
    np.testing.assert_allclose((-a).numpy(), -a.numpy())
    np.testing.assert_allclose((a**2).numpy(), a.numpy() ** 2)


def test_comparison_ops_match_numpy() -> None:
    a = ts.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = ts.tensor([2.0, 2.0])

    np.testing.assert_array_equal((a == b).numpy(), a.numpy() == b.numpy())
    np.testing.assert_array_equal((a != b).numpy(), a.numpy() != b.numpy())
    np.testing.assert_array_equal((a < b).numpy(), a.numpy() < b.numpy())
    np.testing.assert_array_equal((a <= b).numpy(), a.numpy() <= b.numpy())
    np.testing.assert_array_equal((a > b).numpy(), a.numpy() > b.numpy())
    np.testing.assert_array_equal((a >= b).numpy(), a.numpy() >= b.numpy())
    np.testing.assert_array_equal(a.equal(b).numpy(), a.numpy() == b.numpy())
    np.testing.assert_array_equal(ts.not_equal(a, b).numpy(), a.numpy() != b.numpy())
    np.testing.assert_array_equal(ts.less(a, b).numpy(), a.numpy() < b.numpy())
    np.testing.assert_array_equal(a.less_equal(b).numpy(), a.numpy() <= b.numpy())
    np.testing.assert_array_equal(ts.greater(a, b).numpy(), a.numpy() > b.numpy())
    np.testing.assert_array_equal(a.greater_equal(b).numpy(), a.numpy() >= b.numpy())


def test_where_maximum_and_minimum_match_numpy() -> None:
    values = np.array([[1.0, 4.0, -2.0], [7.0, 0.0, 3.0]], dtype=np.float32)
    other = np.array([2.0, 4.0, 1.0], dtype=np.float32)
    condition = np.array([[True, False, True], [False, True, False]])
    x = ts.from_numpy(values)
    y = ts.from_numpy(other)
    mask = ts.from_numpy(condition)

    np.testing.assert_allclose(ts.maximum(x, y).numpy(), np.maximum(values, other))
    np.testing.assert_allclose(x.maximum(y).numpy(), np.maximum(values, other))
    np.testing.assert_allclose(ts.minimum(x, y).numpy(), np.minimum(values, other))
    np.testing.assert_allclose(x.minimum(y).numpy(), np.minimum(values, other))
    np.testing.assert_allclose(ts.where(mask, x, y).numpy(), np.where(condition, values, other))
    np.testing.assert_allclose(mask.where(x, y).numpy(), np.where(condition, values, other))
    np.testing.assert_allclose(ts.clip(x, -1.0, 4.0).numpy(), np.clip(values, -1.0, 4.0))
    np.testing.assert_allclose(ts.clamp(x, -1.0, 4.0).numpy(), np.clip(values, -1.0, 4.0))

    with pytest.raises(Exception, match="condition must have dtype bool"):
        ts.where(x, x, y)


def test_broadcast_error() -> None:
    with pytest.raises(Exception, match="broadcast"):
        _ = ts.ones((2, 3)) + ts.ones((4,))


def test_matmul_reductions_and_activations() -> None:
    a = ts.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = ts.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    np.testing.assert_allclose((a @ b).numpy(), a.numpy() @ b.numpy())
    assert a.sum().item() == pytest.approx(21.0)
    assert a.mean().item() == pytest.approx(3.5)
    assert a.max().item() == pytest.approx(6.0)
    assert a.min().item() == pytest.approx(1.0)

    x = ts.tensor([-1.0, 0.0, 1.0])
    np.testing.assert_allclose(x.relu().numpy(), np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(x.sigmoid().numpy(), 1.0 / (1.0 + np.exp(-x.numpy())), rtol=1e-6)
    np.testing.assert_allclose(x.tanh().numpy(), np.tanh(x.numpy()))
    np.testing.assert_allclose(x.exp().log().numpy(), x.numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(ts.tensor([1.0, 4.0]).sqrt().numpy(), np.array([1.0, 2.0]))
    np.testing.assert_allclose(x.abs().numpy(), np.abs(x.numpy()))
    np.testing.assert_allclose(x.clamp(-0.5, 0.5).numpy(), np.clip(x.numpy(), -0.5, 0.5))
    np.testing.assert_allclose(x.clip(-0.5, 0.5).numpy(), np.clip(x.numpy(), -0.5, 0.5))


def test_blocked_float32_matmul_matches_numpy() -> None:
    rng = np.random.default_rng(123)
    left_np = rng.normal(size=(80, 513)).astype(np.float32)
    right_np = rng.normal(size=(513, 513)).astype(np.float32)

    result = ts.from_numpy(left_np) @ ts.from_numpy(right_np)

    assert result.dtype == "float32"
    np.testing.assert_allclose(result.numpy(), left_np @ right_np, rtol=1e-4, atol=1e-4)


def test_advanced_math_ops_match_numpy() -> None:
    values = np.array([-0.7, -0.2, 0.0, 0.4], dtype=np.float32)
    x = ts.from_numpy(values)

    np.testing.assert_allclose(x.sin().numpy(), np.sin(values), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(ts.cos(x).numpy(), np.cos(values), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(x.tan().numpy(), np.tan(values), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(x.asin().numpy(), np.arcsin(values), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(ts.acos(x).numpy(), np.arccos(values), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(x.atan().numpy(), np.arctan(values), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(ts.log1p(x).numpy(), np.log1p(values), rtol=1e-6, atol=1e-6)

    positive = ts.tensor([0.25, 1.0, 4.0])
    np.testing.assert_allclose(positive.rsqrt().numpy(), 1.0 / np.sqrt(positive.numpy()), rtol=1e-6)


def test_higher_level_math_helpers() -> None:
    values = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]], dtype=np.float32)
    x = ts.from_numpy(values)

    np.testing.assert_allclose(ts.math.square(x).numpy(), values * values)
    np.testing.assert_allclose(ts.math.reciprocal(x).numpy(), 1.0 / values)
    np.testing.assert_allclose(ts.math.variance(x).numpy(), np.var(values), rtol=1e-6)
    np.testing.assert_allclose(ts.math.std(x, axis=1).numpy(), np.std(values, axis=1), rtol=1e-6)
    np.testing.assert_allclose(
        ts.math.norm(x, ord=1, axis=0).numpy(),
        np.linalg.norm(values, ord=1, axis=0),
    )
    np.testing.assert_allclose(ts.math.norm(x).item(), np.linalg.norm(values), rtol=1e-6)

    three_d = np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4)
    y = ts.from_numpy(three_d)
    np.testing.assert_allclose(
        ts.math.variance(y, axis=(0, 2)).numpy(),
        np.var(three_d, axis=(0, 2)),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        ts.math.std(y, axis=(1, 2), keepdims=True).numpy(),
        np.std(three_d, axis=(1, 2), keepdims=True),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        ts.math.norm(y, ord=1, axis=(0, 2)).numpy(),
        np.abs(three_d).sum(axis=(0, 2)),
        rtol=1e-6,
    )


def test_axis_reductions_match_numpy() -> None:
    values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    x = ts.from_numpy(values)

    np.testing.assert_allclose(x.sum(axis=1).numpy(), values.sum(axis=1))
    np.testing.assert_allclose(
        x.sum(axis=0, keepdims=True).numpy(),
        values.sum(axis=0, keepdims=True),
    )
    np.testing.assert_allclose(x.mean(axis=-1).numpy(), values.mean(axis=-1))
    np.testing.assert_allclose(
        x.mean(axis=0, keepdims=True).numpy(),
        values.mean(axis=0, keepdims=True),
    )
    np.testing.assert_allclose(x.max(axis=1).numpy(), values.max(axis=1))
    np.testing.assert_allclose(
        x.min(axis=0, keepdims=True).numpy(),
        values.min(axis=0, keepdims=True),
    )

    with pytest.raises(Exception, match="out of range"):
        x.sum(axis=3)


def test_tuple_axis_reductions_match_numpy() -> None:
    values = np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4)
    x = ts.from_numpy(values)

    for axis in [(0, 2), [1, 2], (-1, 0), ()]:
        np.testing.assert_allclose(x.sum(axis=axis).numpy(), values.sum(axis=tuple(axis)))
        np.testing.assert_allclose(x.mean(axis=axis).numpy(), values.mean(axis=tuple(axis)))
        np.testing.assert_allclose(x.max(axis=axis).numpy(), values.max(axis=tuple(axis)))
        np.testing.assert_allclose(x.min(axis=axis).numpy(), values.min(axis=tuple(axis)))

    np.testing.assert_allclose(
        x.sum(axis=(0, 2), keepdims=True).numpy(),
        values.sum(axis=(0, 2), keepdims=True),
    )
    np.testing.assert_allclose(
        ts.ops.mean(x, axis=(1, 2), keepdims=True).numpy(),
        values.mean(axis=(1, 2), keepdims=True),
    )
    np.testing.assert_allclose(
        ts.ops.max(x, axis=(0, -1), keepdims=True).numpy(),
        values.max(axis=(0, -1), keepdims=True),
    )
    np.testing.assert_allclose(
        ts.ops.min(x, axis=[0, 1]).numpy(),
        values.min(axis=(0, 1)),
    )

    with pytest.raises(Exception, match="duplicate"):
        x.sum(axis=(0, -3))
    with pytest.raises(Exception, match="out of range"):
        x.mean(axis=(0, 4))
    with pytest.raises(Exception, match="axis entries"):
        x.max(axis=(0, "bad"))  # type: ignore[arg-type]


def test_arg_reductions_match_numpy() -> None:
    values = np.array(
        [
            [[1.0, 4.0, 4.0], [7.0, -2.0, 5.0]],
            [[3.0, 3.0, 9.0], [0.0, -5.0, 9.0]],
        ],
        dtype=np.float32,
    )
    x = ts.from_numpy(values)

    assert x.argmax().dtype == "int64"
    assert x.argmin().dtype == "int64"
    assert x.argmax().item() == int(np.argmax(values))
    assert x.argmin().item() == int(np.argmin(values))
    np.testing.assert_array_equal(x.argmax(axis=0).numpy(), np.argmax(values, axis=0))
    np.testing.assert_array_equal(x.argmin(axis=1).numpy(), np.argmin(values, axis=1))
    np.testing.assert_array_equal(
        x.argmax(axis=-1, keepdims=True).numpy(),
        np.argmax(values, axis=-1, keepdims=True),
    )
    np.testing.assert_array_equal(
        ts.ops.argmin(x, axis=2, keepdims=True).numpy(),
        np.argmin(values, axis=2, keepdims=True),
    )
    np.testing.assert_array_equal(
        ts.argmax(x, keepdims=True).numpy(),
        np.argmax(values, keepdims=True),
    )

    ties = ts.tensor([2.0, 5.0, 5.0, 1.0])
    assert ties.argmax().item() == 1

    with pytest.raises(Exception, match="out of range"):
        x.argmax(axis=3)
    with pytest.raises(Exception, match="axis must be None or an int"):
        x.argmin(axis=(0, 1))  # type: ignore[arg-type]
    with pytest.raises(Exception, match="empty"):
        ts.tensor([], dtype="float32").argmax()
    with pytest.raises(Exception, match="empty reduction axes"):
        ts.empty((2, 0)).argmin(axis=1)


def test_cast_concat_and_stack_match_numpy() -> None:
    x = ts.tensor([1.2, 2.8, -3.1])
    as_int = x.astype("int32")
    assert as_int.dtype == "int32"
    np.testing.assert_array_equal(as_int.numpy(), np.array([1, 2, -3], dtype=np.int32))
    np.testing.assert_allclose(as_int.to("float64").numpy(), np.array([1.0, 2.0, -3.0]))

    a_np = np.array([[1.0, 2.0]], dtype=np.float32)
    b_np = np.array([[3.0, 4.0]], dtype=np.float32)
    a = ts.from_numpy(a_np)
    b = ts.from_numpy(b_np)

    np.testing.assert_allclose(
        ts.concat([a, b], axis=0).numpy(),
        np.concatenate([a_np, b_np], axis=0),
    )
    np.testing.assert_allclose(
        ts.concat([a, b], axis=1).numpy(),
        np.concatenate([a_np, b_np], axis=1),
    )
    np.testing.assert_allclose(ts.stack([a, b], axis=0).numpy(), np.stack([a_np, b_np], axis=0))
    np.testing.assert_allclose(ts.stack([a, b], axis=-1).numpy(), np.stack([a_np, b_np], axis=-1))

    with pytest.raises(Exception, match="same dtype"):
        ts.concat([a, ts.tensor([[1, 2]])])
    with pytest.raises(Exception, match="identical shapes"):
        ts.stack([a, ts.ones((2, 2))])


def test_conv2d_matches_numpy_reference() -> None:
    x_np = np.arange(1, 19, dtype=np.float32).reshape(1, 2, 3, 3)
    w_np = np.array(
        [
            [[[1.0, 0.0], [0.0, -1.0]], [[0.5, 0.25], [-0.5, 1.0]]],
            [[[-1.0, 2.0], [0.5, 0.0]], [[1.0, -1.0], [0.0, 0.5]]],
        ],
        dtype=np.float32,
    )
    b_np = np.array([0.5, -1.0], dtype=np.float32)

    actual = ts.conv2d(ts.from_numpy(x_np), ts.from_numpy(w_np), ts.from_numpy(b_np))
    expected = _numpy_conv2d(x_np, w_np, b_np)

    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_conv2d_stride_padding_and_dilation() -> None:
    x_np = np.arange(1, 26, dtype=np.float32).reshape(1, 1, 5, 5)
    w_np = np.array([[[[1.0, -1.0], [2.0, 0.5]]]], dtype=np.float32)

    actual = ts.conv2d(
        ts.from_numpy(x_np),
        ts.from_numpy(w_np),
        stride=(2, 2),
        padding=(1, 1),
        dilation=(2, 2),
    )
    expected = _numpy_conv2d(x_np, w_np, stride=(2, 2), padding=(1, 1), dilation=(2, 2))

    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_conv2d_shape_errors_are_clear() -> None:
    with pytest.raises(Exception, match="NCHW"):
        ts.conv2d(ts.ones((1, 3, 3)), ts.ones((1, 1, 2, 2)))
    with pytest.raises(Exception, match="channel mismatch"):
        ts.conv2d(ts.ones((1, 2, 3, 3)), ts.ones((1, 1, 2, 2)))


def test_pool2d_matches_numpy_reference() -> None:
    x_np = np.arange(1, 17, dtype=np.float32).reshape(1, 1, 4, 4)
    x = ts.from_numpy(x_np)

    np.testing.assert_allclose(
        ts.max_pool2d(x, 2).numpy(),
        _numpy_max_pool2d(x_np, (2, 2)),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        ts.avg_pool2d(x, 2).numpy(),
        _numpy_avg_pool2d(x_np, (2, 2)),
        rtol=1e-6,
    )


def test_pool2d_stride_padding_and_dilation() -> None:
    x_np = np.arange(1, 26, dtype=np.float32).reshape(1, 1, 5, 5)
    x = ts.from_numpy(x_np)

    np.testing.assert_allclose(
        ts.max_pool2d(x, kernel_size=(2, 2), stride=(2, 1), padding=1, dilation=2).numpy(),
        _numpy_max_pool2d(x_np, (2, 2), stride=(2, 1), padding=(1, 1), dilation=(2, 2)),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        ts.avg_pool2d(x, kernel_size=(3, 3), stride=2, padding=1).numpy(),
        _numpy_avg_pool2d(x_np, (3, 3), stride=(2, 2), padding=(1, 1)),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        ts.avg_pool2d(x, kernel_size=3, stride=2, padding=1, count_include_pad=True).numpy(),
        _numpy_avg_pool2d(
            x_np,
            (3, 3),
            stride=(2, 2),
            padding=(1, 1),
            count_include_pad=True,
        ),
        rtol=1e-6,
    )


def test_pool2d_shape_errors_are_clear() -> None:
    with pytest.raises(Exception, match="NCHW"):
        ts.max_pool2d(ts.ones((1, 3, 3)), 2)
    with pytest.raises(Exception, match="kernel_size"):
        ts.avg_pool2d(ts.ones((1, 1, 3, 3)), 0)


def test_views() -> None:
    x = ts.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    np.testing.assert_allclose(x.reshape((3, 2)).numpy(), x.numpy().reshape(3, 2))
    np.testing.assert_allclose(x.flatten().numpy(), x.numpy().flatten())
    np.testing.assert_allclose(x.T.numpy(), x.numpy().T)
