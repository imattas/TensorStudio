from __future__ import annotations

import numpy as np
import pytest
import tensorstudio as ts
from tensorstudio.errors import AutogradError


def _finite_difference(values: np.ndarray, objective, eps: float = 1e-5) -> np.ndarray:
    expected = np.zeros_like(values, dtype=np.float64)
    for index in np.ndindex(values.shape):
        plus = values.copy()
        minus = values.copy()
        plus[index] += eps
        minus[index] -= eps
        y_plus = objective(ts.tensor(plus.tolist(), dtype="float64")).item()
        y_minus = objective(ts.tensor(minus.tolist(), dtype="float64")).item()
        expected[index] = (y_plus - y_minus) / (2.0 * eps)
    return expected


def _assert_grad_matches(values: np.ndarray, objective) -> None:
    x = ts.tensor(values.tolist(), dtype="float64", requires_grad=True)
    y = objective(x)
    y.backward()

    expected = _finite_difference(values.astype(np.float64), objective)
    np.testing.assert_allclose(x.grad.numpy(), expected, rtol=2e-3, atol=2e-3)


def test_scalar_autograd_square_sum() -> None:
    x = ts.tensor([1.0, 2.0, 3.0], requires_grad=True)

    loss = (x * x).sum()
    loss.backward()

    np.testing.assert_allclose(x.grad.numpy(), np.array([2.0, 4.0, 6.0]), rtol=1e-6)


def test_broadcast_autograd() -> None:
    x = ts.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = ts.tensor([10.0, 20.0], requires_grad=True)

    y = (x + b).sum()
    y.backward()

    np.testing.assert_allclose(x.grad.numpy(), np.ones((2, 2)), rtol=1e-6)
    np.testing.assert_allclose(b.grad.numpy(), np.array([2.0, 2.0]), rtol=1e-6)


def test_backward_accepts_explicit_gradient_for_non_scalar() -> None:
    x = ts.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x * 2.0

    y.backward(ts.tensor([1.0, 0.5, 0.25]))

    np.testing.assert_allclose(x.grad.numpy(), np.array([2.0, 1.0, 0.5]), rtol=1e-6)


def test_matmul_autograd() -> None:
    x = ts.tensor([[1.0, 2.0]], requires_grad=True)
    w = ts.tensor([[3.0], [4.0]], requires_grad=True)

    y = (x @ w).sum()
    y.backward()

    np.testing.assert_allclose(x.grad.numpy(), np.array([[3.0, 4.0]]), rtol=1e-6)
    np.testing.assert_allclose(w.grad.numpy(), np.array([[1.0], [2.0]]), rtol=1e-6)


def test_activation_autograd() -> None:
    x = ts.tensor([-1.0, 0.0, 1.0], requires_grad=True)

    y = (x.sigmoid() + x.tanh() + x.exp()).sum()
    y.backward()

    xv = x.numpy()
    sig = 1.0 / (1.0 + np.exp(-xv))
    expected = sig * (1.0 - sig) + (1.0 - np.tanh(xv) ** 2) + np.exp(xv)
    np.testing.assert_allclose(x.grad.numpy(), expected, rtol=1e-5)


def test_additional_unary_and_reduction_autograd() -> None:
    x = ts.tensor([-2.0, -1.0, 4.0], requires_grad=True)
    y = (x.abs() + (x * x).sqrt() + x.clamp(-1.5, 3.0)).sum()
    y.backward()

    np.testing.assert_allclose(x.grad.numpy(), np.array([-2.0, -1.0, 2.0]), rtol=1e-6)

    z = ts.tensor([1.0, 3.0, 3.0, -2.0], requires_grad=True)
    (z.max() + z.min()).backward()
    np.testing.assert_allclose(z.grad.numpy(), np.array([0.0, 0.5, 0.5, 1.0]), rtol=1e-6)


def test_axis_reduction_autograd() -> None:
    x = ts.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    weighted = x.sum(axis=1) * ts.tensor([1.0, 2.0])
    weighted.sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))

    y = ts.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    y.mean(axis=0).sum().backward()
    np.testing.assert_allclose(y.grad.numpy(), np.full((2, 3), 0.5), rtol=1e-6)

    z = ts.tensor([[1.0, 3.0, 3.0], [2.0, 2.0, 0.0]], requires_grad=True)
    z.max(axis=1).sum().backward()
    np.testing.assert_allclose(z.grad.numpy(), np.array([[0.0, 0.5, 0.5], [0.5, 0.5, 0.0]]))


def test_tuple_axis_reduction_autograd() -> None:
    values = np.arange(1, 25, dtype=np.float64).reshape(2, 3, 4)

    x = ts.tensor(values.tolist(), dtype="float64", requires_grad=True)
    (x.sum(axis=(0, 2)) * ts.tensor([1.0, 2.0, 3.0], dtype="float64")).sum().backward()
    expected = np.broadcast_to(np.array([1.0, 2.0, 3.0]).reshape(1, 3, 1), values.shape)
    np.testing.assert_allclose(x.grad.numpy(), expected, rtol=1e-6)

    y = ts.tensor(values.tolist(), dtype="float64", requires_grad=True)
    y.mean(axis=(1, 2)).sum().backward()
    np.testing.assert_allclose(y.grad.numpy(), np.full(values.shape, 1.0 / 12.0), rtol=1e-6)

    z = ts.tensor(
        [[[1.0, 4.0], [4.0, 2.0]], [[3.0, 3.0], [0.0, 5.0]]],
        dtype="float64",
        requires_grad=True,
    )
    z.max(axis=(1, 2)).sum().backward()
    np.testing.assert_allclose(
        z.grad.numpy(),
        np.array([[[0.0, 0.5], [0.5, 0.0]], [[0.0, 0.0], [0.0, 1.0]]]),
        rtol=1e-6,
    )


def test_where_maximum_minimum_autograd() -> None:
    x = ts.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = ts.tensor([[10.0, 20.0], [30.0, 40.0]], requires_grad=True)
    mask = ts.tensor([[True, False], [False, True]])
    weights = ts.tensor([[1.0, 2.0], [3.0, 4.0]])

    (ts.where(mask, x, y) * weights).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), np.array([[1.0, 0.0], [0.0, 4.0]]), rtol=1e-6)
    np.testing.assert_allclose(y.grad.numpy(), np.array([[0.0, 2.0], [3.0, 0.0]]), rtol=1e-6)

    left = ts.tensor([1.0, 3.0, 3.0], requires_grad=True)
    right = ts.tensor([2.0, 3.0, 1.0], requires_grad=True)
    ts.maximum(left, right).sum().backward()
    np.testing.assert_allclose(left.grad.numpy(), np.array([0.0, 0.5, 1.0]), rtol=1e-6)
    np.testing.assert_allclose(right.grad.numpy(), np.array([1.0, 0.5, 0.0]), rtol=1e-6)

    a = ts.tensor([1.0, 3.0, 3.0], requires_grad=True)
    b = ts.tensor([2.0, 3.0, 1.0], requires_grad=True)
    ts.minimum(a, b).sum().backward()
    np.testing.assert_allclose(a.grad.numpy(), np.array([1.0, 0.5, 0.0]), rtol=1e-6)
    np.testing.assert_allclose(b.grad.numpy(), np.array([0.0, 0.5, 1.0]), rtol=1e-6)


def test_cast_concat_stack_autograd() -> None:
    x = ts.tensor([1.0, 2.0], requires_grad=True)
    x.astype("float64").sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), np.ones(2), rtol=1e-6)

    a = ts.tensor([[1.0, 2.0]], requires_grad=True)
    b = ts.tensor([[3.0, 4.0]], requires_grad=True)
    weights = ts.tensor([[1.0, 2.0], [3.0, 4.0]])
    (ts.concat([a, b], axis=0) * weights).sum().backward()
    np.testing.assert_allclose(a.grad.numpy(), np.array([[1.0, 2.0]]), rtol=1e-6)
    np.testing.assert_allclose(b.grad.numpy(), np.array([[3.0, 4.0]]), rtol=1e-6)

    c = ts.tensor([1.0, 2.0], requires_grad=True)
    d = ts.tensor([3.0, 4.0], requires_grad=True)
    (ts.stack([c, d], axis=1) * ts.tensor([[1.0, 2.0], [3.0, 4.0]])).sum().backward()
    np.testing.assert_allclose(c.grad.numpy(), np.array([1.0, 3.0]), rtol=1e-6)
    np.testing.assert_allclose(d.grad.numpy(), np.array([2.0, 4.0]), rtol=1e-6)


def test_checkpoint_recomputes_forward_and_preserves_gradients() -> None:
    calls = {"count": 0}
    x = ts.tensor([1.0, -2.0, 3.0], requires_grad=True)

    def block(input: ts.Tensor) -> ts.Tensor:
        calls["count"] += 1
        return ((input * input) + 1.0).sum()

    loss = ts.checkpoint(block, x)

    assert calls["count"] == 1
    assert loss.requires_grad is True
    loss.backward()

    assert calls["count"] == 2
    np.testing.assert_allclose(x.grad.numpy(), np.array([2.0, -4.0, 6.0]), rtol=1e-6)


def test_checkpoint_accepts_explicit_backward_gradient() -> None:
    x = ts.tensor([1.0, 2.0, 3.0], requires_grad=True)
    weights = ts.tensor([1.0, 0.5, 0.25])

    y = ts.checkpoint(lambda input: input * input, x)
    y.backward(weights)

    np.testing.assert_allclose(x.grad.numpy(), np.array([2.0, 2.0, 1.5]), rtol=1e-6)


def test_checkpoint_returns_plain_output_without_grad_inputs() -> None:
    calls = {"count": 0}
    x = ts.tensor([1.0, 2.0, 3.0])

    def block(input: ts.Tensor) -> ts.Tensor:
        calls["count"] += 1
        return input * 2.0

    y = ts.checkpoint(block, x)

    assert calls["count"] == 1
    assert y.requires_grad is False
    np.testing.assert_allclose(y.numpy(), np.array([2.0, 4.0, 6.0]), rtol=1e-6)


def test_backward_detects_in_place_parent_mutation() -> None:
    x = ts.tensor([1.0, 2.0, 3.0], requires_grad=True)
    loss = (x * x).sum()

    x._assign(ts.ones(x.shape))

    with pytest.raises(AutogradError, match="modified in-place"):
        loss.backward()


def test_backward_detects_in_place_mutation_through_shared_storage() -> None:
    x = ts.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    row = x[0]
    loss = (row * row).sum()

    x._assign(ts.zeros(x.shape))

    with pytest.raises(AutogradError, match="modified in-place"):
        loss.backward()


def test_optimizer_update_after_backward_is_still_allowed() -> None:
    parameter = ts.tensor([1.0, 2.0, 3.0], requires_grad=True)
    loss = (parameter * parameter).sum()
    loss.backward()

    optimizer = ts.optim.SGD([parameter], lr=0.1)
    optimizer.step()

    np.testing.assert_allclose(parameter.numpy(), np.array([0.8, 1.6, 2.4]), rtol=1e-6)


def test_reshape_transpose_autograd() -> None:
    x = ts.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    y = (x.reshape((4,)) * 2).sum()
    y.backward()

    np.testing.assert_allclose(x.grad.numpy(), np.full((2, 2), 2.0), rtol=1e-6)

    w = ts.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    z = (w.T * 3).sum()
    z.backward()
    np.testing.assert_allclose(w.grad.numpy(), np.full((2, 2), 3.0), rtol=1e-6)


def test_finite_difference_selected_ops() -> None:
    values = np.array([[0.4, -0.7], [1.2, -1.5]], dtype=np.float64)

    _assert_grad_matches(values, lambda x: (x + 2.0).mean())
    _assert_grad_matches(values, lambda x: (x * x).mean())
    _assert_grad_matches(values, lambda x: (x / 3.0).mean())
    _assert_grad_matches(values, lambda x: x.mean())
    _assert_grad_matches(values, lambda x: x.relu().mean())
    _assert_grad_matches(values, lambda x: x.sigmoid().mean())
    _assert_grad_matches(values, lambda x: x.tanh().mean())
    _assert_grad_matches(values, lambda x: x.sum(axis=1).mean())
    _assert_grad_matches(values, lambda x: x.mean(axis=0).sum())


def test_finite_difference_advanced_math_ops() -> None:
    values = np.array([[0.2, -0.3], [0.5, -0.6]], dtype=np.float64)
    positive = np.array([[0.2, 0.8], [1.4, 2.2]], dtype=np.float64)

    _assert_grad_matches(values, lambda x: (x.sin() + x.cos() + x.tan() + x.atan()).mean())
    _assert_grad_matches(values, lambda x: (x.asin() + x.acos()).mean())
    _assert_grad_matches(positive, lambda x: (x.log1p() + x.rsqrt()).mean())


def test_finite_difference_matmul() -> None:
    values = np.array([[0.4, -0.7], [1.2, -1.5]], dtype=np.float64)
    weight = ts.tensor([[1.0, -2.0], [0.5, 3.0]], dtype="float64")

    _assert_grad_matches(values, lambda x: (x @ weight).mean())


def test_finite_difference_conv2d() -> None:
    input_values = np.array(
        [[[[0.2, -0.4, 0.7], [1.0, -1.2, 0.5], [0.3, 0.8, -0.6]]]],
        dtype=np.float64,
    )
    weight_values = np.array([[[[0.5, -0.25], [1.5, 0.75]]]], dtype=np.float64)
    bias_values = np.array([0.1], dtype=np.float64)

    weight = ts.tensor(weight_values.tolist(), dtype="float64")
    bias = ts.tensor(bias_values.tolist(), dtype="float64")
    _assert_grad_matches(
        input_values,
        lambda x: ts.conv2d(x, weight, bias, padding=1).mean(),
    )

    fixed_input = ts.tensor(input_values.tolist(), dtype="float64")
    _assert_grad_matches(
        weight_values,
        lambda w: ts.conv2d(fixed_input, w, bias, padding=1).mean(),
    )

    _assert_grad_matches(
        bias_values,
        lambda b: ts.conv2d(fixed_input, weight, b, padding=1).mean(),
    )


def test_finite_difference_pool2d() -> None:
    values = np.array(
        [
            [
                [
                    [0.2, -0.4, 0.7, 1.1],
                    [1.0, -1.2, 0.5, 0.9],
                    [0.3, 0.8, -0.6, 1.4],
                    [1.7, -0.1, 0.6, -0.9],
                ]
            ]
        ],
        dtype=np.float64,
    )

    _assert_grad_matches(values, lambda x: ts.avg_pool2d(x, kernel_size=2).mean())
    _assert_grad_matches(values, lambda x: ts.max_pool2d(x, kernel_size=2).mean())
