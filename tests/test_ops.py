from __future__ import annotations

import numpy as np
import pytest
import tensorstudio as ts


def test_elementwise_ops_match_numpy() -> None:
    a = ts.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = ts.tensor([10.0, 20.0])

    np.testing.assert_allclose((a + b).numpy(), np.array([[11.0, 22.0], [13.0, 24.0]]))
    np.testing.assert_allclose((a - 1).numpy(), np.array([[0.0, 1.0], [2.0, 3.0]]))
    np.testing.assert_allclose((a * 2).numpy(), np.array([[2.0, 4.0], [6.0, 8.0]]))
    np.testing.assert_allclose((a / 2).numpy(), np.array([[0.5, 1.0], [1.5, 2.0]]))
    np.testing.assert_allclose((-a).numpy(), -a.numpy())
    np.testing.assert_allclose((a**2).numpy(), a.numpy() ** 2)


def test_broadcast_error() -> None:
    with pytest.raises(Exception, match="broadcast"):
        _ = ts.ones((2, 3)) + ts.ones((4,))


def test_matmul_reductions_and_activations() -> None:
    a = ts.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = ts.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    np.testing.assert_allclose((a @ b).numpy(), a.numpy() @ b.numpy())
    assert a.sum().item() == pytest.approx(21.0)
    assert a.mean().item() == pytest.approx(3.5)

    x = ts.tensor([-1.0, 0.0, 1.0])
    np.testing.assert_allclose(x.relu().numpy(), np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(x.sigmoid().numpy(), 1.0 / (1.0 + np.exp(-x.numpy())))
    np.testing.assert_allclose(x.tanh().numpy(), np.tanh(x.numpy()))
    np.testing.assert_allclose(x.exp().log().numpy(), x.numpy(), rtol=1e-6, atol=1e-6)


def test_views() -> None:
    x = ts.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    np.testing.assert_allclose(x.reshape((3, 2)).numpy(), x.numpy().reshape(3, 2))
    np.testing.assert_allclose(x.flatten().numpy(), x.numpy().flatten())
    np.testing.assert_allclose(x.T.numpy(), x.numpy().T)
