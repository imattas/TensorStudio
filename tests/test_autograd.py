from __future__ import annotations

import numpy as np
import tensorstudio as ts


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


def test_finite_difference_matmul() -> None:
    values = np.array([[0.4, -0.7], [1.2, -1.5]], dtype=np.float64)
    weight = ts.tensor([[1.0, -2.0], [0.5, 3.0]], dtype="float64")

    _assert_grad_matches(values, lambda x: (x @ weight).mean())
