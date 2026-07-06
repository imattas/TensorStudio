from __future__ import annotations

import numpy as np
import tensorstudio as ts


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


def test_reshape_transpose_autograd() -> None:
    x = ts.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    y = (x.reshape((4,)) * 2).sum()
    y.backward()

    np.testing.assert_allclose(x.grad.numpy(), np.full((2, 2), 2.0), rtol=1e-6)

    w = ts.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    z = (w.T * 3).sum()
    z.backward()
    np.testing.assert_allclose(w.grad.numpy(), np.full((2, 2), 3.0), rtol=1e-6)
