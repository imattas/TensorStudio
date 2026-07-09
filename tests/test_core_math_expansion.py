from __future__ import annotations

import numpy as np
import pytest
import tensorstudio as ts


def _numpy_logsumexp(values: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    result = np.log(np.sum(np.exp(values - max_values), axis=axis, keepdims=True)) + max_values
    if keepdims:
        return result
    if axis is None:
        return np.asarray(result.squeeze())
    return np.squeeze(result, axis=axis)


def test_stable_softmax_log_softmax_and_logsumexp_match_numpy() -> None:
    values = np.array([[1000.0, 1001.0, 999.0], [-1000.0, -999.0, -1001.0]], dtype=np.float64)
    x = ts.from_numpy(values)

    shifted = values - values.max(axis=1, keepdims=True)
    expected_softmax = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    expected_log_softmax = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))

    np.testing.assert_allclose(
        ts.softmax(x, axis=1).numpy(), expected_softmax, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(x.softmax(axis=-1).numpy(), expected_softmax, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(
        ts.log_softmax(x, axis=1).numpy(), expected_log_softmax, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        ts.logsumexp(x).numpy(), _numpy_logsumexp(values), rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        x.logsumexp(axis=1, keepdims=True).numpy(),
        _numpy_logsumexp(values, axis=1, keepdims=True),
        rtol=1e-7,
        atol=1e-7,
    )


def test_softmax_and_logsumexp_autograd_are_finite() -> None:
    x = ts.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]], dtype="float64", requires_grad=True)
    loss = (x.softmax(axis=1) * x.logsumexp(axis=1, keepdims=True)).sum()
    loss.backward()

    assert x.grad is not None
    assert np.isfinite(x.grad.numpy()).all()
    assert x.grad.shape == x.shape


def test_batched_matmul_matches_numpy_and_has_gradients() -> None:
    left_np = np.arange(1, 13, dtype=np.float64).reshape(2, 2, 3) / 10.0
    right_np = np.arange(1, 25, dtype=np.float64).reshape(2, 3, 4) / 20.0
    left = ts.tensor(left_np.tolist(), dtype="float64", requires_grad=True)
    right = ts.tensor(right_np.tolist(), dtype="float64", requires_grad=True)

    actual = left @ right
    np.testing.assert_allclose(actual.numpy(), np.matmul(left_np, right_np), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(
        ts.bmm(left, right).numpy(), np.matmul(left_np, right_np), rtol=1e-9, atol=1e-9
    )

    actual.sum().backward()
    expected_left_grad = np.repeat(right_np.sum(axis=2)[:, np.newaxis, :], left_np.shape[1], axis=1)
    expected_right_grad = np.repeat(
        left_np.sum(axis=1)[:, :, np.newaxis], right_np.shape[2], axis=2
    )
    np.testing.assert_allclose(left.grad.numpy(), expected_left_grad, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(right.grad.numpy(), expected_right_grad, rtol=1e-9, atol=1e-9)


def test_native_statistical_reductions_match_numpy() -> None:
    values = np.array([[1.0, 2.0, 5.0], [2.0, 4.0, 8.0]], dtype=np.float64)
    x = ts.from_numpy(values)

    np.testing.assert_allclose(x.var().numpy(), np.var(values), rtol=1e-9)
    np.testing.assert_allclose(ts.var(x, axis=1).numpy(), np.var(values, axis=1), rtol=1e-9)
    np.testing.assert_allclose(
        ts.variance(x, axis=0, correction=1).numpy(), np.var(values, axis=0, ddof=1), rtol=1e-9
    )
    np.testing.assert_allclose(x.std(axis=1).numpy(), np.std(values, axis=1), rtol=1e-9)
    np.testing.assert_allclose(x.norm().numpy(), np.linalg.norm(values), rtol=1e-9)
    np.testing.assert_allclose(
        x.norm(ord=1, axis=0).numpy(), np.linalg.norm(values, ord=1, axis=0), rtol=1e-9
    )


def test_boolean_reductions_match_numpy() -> None:
    values = np.array([[True, True, False], [True, True, True]], dtype=bool)
    x = ts.from_numpy(values)

    assert ts.all(x).item() is False
    assert ts.any(x).item() is True
    np.testing.assert_array_equal(x.all(axis=1).numpy(), values.all(axis=1))
    np.testing.assert_array_equal(
        x.any(axis=0, keepdims=True).numpy(), values.any(axis=0, keepdims=True)
    )
    np.testing.assert_array_equal(ts.all(x, axis=(0, 1)).numpy(), values.all(axis=(0, 1)))


def test_random_distributions_are_seeded_and_valid() -> None:
    uniform_a = ts.uniform((32,), low=-2.0, high=3.0, seed=7)
    uniform_b = ts.uniform((32,), low=-2.0, high=3.0, seed=7)
    np.testing.assert_allclose(uniform_a.numpy(), uniform_b.numpy())
    assert np.all(uniform_a.numpy() >= -2.0)
    assert np.all(uniform_a.numpy() < 3.0)

    normal_a = ts.normal((8,), mean=5.0, stddev=0.25, seed=11)
    normal_b = ts.normal((8,), mean=5.0, stddev=0.25, seed=11)
    np.testing.assert_allclose(normal_a.numpy(), normal_b.numpy())

    ints = ts.randint((128,), low=2, high=6, seed=3)
    assert ints.dtype == "int64"
    assert ints.numpy().min() >= 2
    assert ints.numpy().max() < 6

    samples = ts.bernoulli((64,), probability=0.25, seed=5)
    assert samples.dtype == "bool"
    assert set(samples.numpy().tolist()) <= {False, True}

    with pytest.raises(Exception, match="high > low"):
        ts.uniform((2,), low=1.0, high=1.0)
    with pytest.raises(Exception, match="int32 or int64"):
        ts.randint((2,), low=0, high=2, dtype="float32")


def test_einsum_supported_subset_matches_numpy() -> None:
    matrix = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    other = np.arange(1, 13, dtype=np.float32).reshape(3, 4)
    batch_left = np.arange(1, 13, dtype=np.float32).reshape(2, 2, 3)
    batch_right = np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4)

    x = ts.from_numpy(matrix)
    y = ts.from_numpy(other)
    bx = ts.from_numpy(batch_left)
    by = ts.from_numpy(batch_right)

    np.testing.assert_allclose(
        ts.einsum("ij,jk->ik", x, y).numpy(), np.einsum("ij,jk->ik", matrix, other)
    )
    np.testing.assert_allclose(
        ts.einsum("bij,bjk->bik", bx, by).numpy(),
        np.einsum("bij,bjk->bik", batch_left, batch_right),
    )
    np.testing.assert_allclose(
        ts.einsum("ij,ij->", x, x).numpy(), np.einsum("ij,ij->", matrix, matrix)
    )
    np.testing.assert_allclose(ts.einsum("ij->ji", x).numpy(), np.einsum("ij->ji", matrix))
    np.testing.assert_allclose(ts.einsum("ij->i", x).numpy(), np.einsum("ij->i", matrix))
    np.testing.assert_allclose(ts.einsum("ij->j", x).numpy(), np.einsum("ij->j", matrix))

    with pytest.raises(ValueError, match="unsupported"):
        ts.einsum("ij->ij", x)
