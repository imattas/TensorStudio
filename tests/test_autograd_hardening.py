from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import tensorstudio as ts


def _finite_difference(
    values: np.ndarray,
    objective: Callable[[ts.Tensor], ts.Tensor],
    eps: float = 1e-5,
) -> np.ndarray:
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


def _assert_grad_matches(values: np.ndarray, objective: Callable[[ts.Tensor], ts.Tensor]) -> None:
    x = ts.tensor(values.tolist(), dtype="float64", requires_grad=True)
    objective(x).backward()

    expected = _finite_difference(values.astype(np.float64), objective)
    np.testing.assert_allclose(x.grad.numpy(), expected, rtol=3e-3, atol=3e-3)


def test_backward_releases_graph_unless_retained() -> None:
    x = ts.tensor([1.0, 2.0], requires_grad=True)
    loss = (x * x).sum()

    loss.backward()
    np.testing.assert_allclose(x.grad.numpy(), np.array([2.0, 4.0]))

    with pytest.raises(Exception, match="retain_graph=True"):
        loss.backward()


def test_retain_graph_reuses_graph_and_accumulates_leaf_gradients() -> None:
    x = ts.tensor([1.0, 2.0], requires_grad=True)
    loss = (x * x).sum()

    loss.backward(retain_graph=True)
    np.testing.assert_allclose(x.grad.numpy(), np.array([2.0, 4.0]))

    loss.backward()
    np.testing.assert_allclose(x.grad.numpy(), np.array([4.0, 8.0]))

    with pytest.raises(Exception, match="already been freed"):
        loss.backward()


def test_leaf_history_and_detach_controls() -> None:
    x = ts.tensor([1.0, 2.0], requires_grad=True)
    y = x * 2.0

    assert x.is_leaf is True
    assert y.is_leaf is False

    y.clear_history()
    assert y.is_leaf is True
    y.sum().backward()
    assert x.grad is None

    z = x * 3.0
    z.detach_()
    assert z.requires_grad is False
    assert z.is_leaf is True
    z.sum().backward()
    assert x.grad is None


def test_safe_inplace_methods_require_no_grad_for_grad_tensors() -> None:
    x = ts.ones((2,), requires_grad=True)

    with pytest.raises(Exception, match="in-place"):
        x.add_(ts.ones((2,)))
    with pytest.raises(Exception, match="in-place"):
        x.zero_()

    with ts.no_grad():
        x.add_(ts.ones((2,)), alpha=2.0)
        x.fill_(5.0)

    np.testing.assert_allclose(x.numpy(), np.array([5.0, 5.0]))
    assert x.requires_grad is True

    y = ts.ones((2,))
    y.zero_().fill_(3.0).add_(ts.ones((2,)), alpha=0.5)
    np.testing.assert_allclose(y.numpy(), np.array([3.5, 3.5]))


def test_non_scalar_backward_through_new_math_ops() -> None:
    x = ts.tensor([[0.2, -0.4, 0.7], [1.0, -1.2, 0.5]], dtype="float64", requires_grad=True)
    y = x.softmax(axis=1)

    y.backward(ts.tensor([[1.0, 0.5, -0.25], [0.25, -0.5, 1.0]], dtype="float64"))

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert np.isfinite(x.grad.numpy()).all()


def test_finite_difference_stable_probability_ops() -> None:
    values = np.array([[0.2, -0.4, 0.7], [1.0, -1.2, 0.5]], dtype=np.float64)
    weights = ts.tensor([[1.0, -0.5, 0.25], [0.25, 0.75, -1.0]], dtype="float64")

    _assert_grad_matches(values, lambda x: x.logsumexp(axis=1).sum())
    _assert_grad_matches(values, lambda x: (x.softmax(axis=1) * weights).sum())
    _assert_grad_matches(values, lambda x: (x.log_softmax(axis=1) * weights).sum())


def test_finite_difference_statistics_and_norm() -> None:
    values = np.array([[0.5, 1.5, 2.0], [1.0, 2.5, 4.0]], dtype=np.float64)

    _assert_grad_matches(values, lambda x: x.var(axis=1).sum())
    _assert_grad_matches(values, lambda x: x.std(axis=1).sum())
    _assert_grad_matches(values, lambda x: x.norm(ord=2, axis=1).sum())
    _assert_grad_matches(values, lambda x: ts.math.variance(x, axis=(0, 1)))


def test_finite_difference_batched_matmul() -> None:
    values = np.array(
        [[[0.2, -0.4], [0.7, 1.0]], [[-1.2, 0.5], [0.3, -0.8]]],
        dtype=np.float64,
    )
    right = ts.tensor(
        [[[0.5, -0.25], [1.5, 0.75]], [[-0.5, 0.25], [0.8, -1.1]]],
        dtype="float64",
    )

    _assert_grad_matches(values, lambda x: (x @ right).mean())
