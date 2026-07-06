from __future__ import annotations

import numpy as np
import pytest
import tensorstudio as ts


def test_tensor_creation_and_properties() -> None:
    x = ts.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    assert x.shape == (2, 2)
    assert x.strides == (2, 1)
    assert x.device == "cpu"
    assert x.ndim == 2
    assert x.size == 4
    assert x.dtype == "float32"
    assert x.requires_grad is True
    assert x.is_contiguous is True
    assert x.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_dtype_creation() -> None:
    assert ts.tensor([1, 2, 3]).dtype == "int64"
    assert ts.tensor([True, False]).dtype == "bool"
    assert ts.tensor([1, 2, 3], dtype="int32").dtype == "int32"
    assert ts.tensor([1.0], dtype="float64").dtype == "float64"


def test_creation_helpers() -> None:
    assert ts.empty((2, 3)).shape == (2, 3)
    np.testing.assert_allclose(ts.zeros((2, 2)).numpy(), np.zeros((2, 2), dtype=np.float32))
    np.testing.assert_allclose(ts.ones((2,)).numpy(), np.ones((2,), dtype=np.float32))
    np.testing.assert_allclose(ts.full((3,), 7).numpy(), np.full((3,), 7, dtype=np.float32))
    np.testing.assert_allclose(ts.arange(4).numpy(), np.arange(4, dtype=np.float32))
    np.testing.assert_allclose(ts.eye(3).numpy(), np.eye(3, dtype=np.float32))
    np.testing.assert_allclose(ts.eye(2, 3).numpy(), np.eye(2, 3, dtype=np.float32))
    np.testing.assert_allclose(ts.linspace(0, 1, 5).numpy(), np.linspace(0, 1, 5, dtype=np.float32))


def test_rand_and_manual_seed_are_deterministic() -> None:
    ts.manual_seed(123)
    a = ts.rand((2, 2))
    ts.manual_seed(123)
    b = ts.rand((2, 2))

    np.testing.assert_allclose(a.numpy(), b.numpy())


def test_randn_seed_is_deterministic() -> None:
    a = ts.randn((2, 2), seed=123)
    b = ts.randn((2, 2), seed=123)

    np.testing.assert_allclose(a.numpy(), b.numpy())


def test_clone_detach_and_zero_grad() -> None:
    x = ts.tensor([1.0, 2.0], requires_grad=True)
    cloned = x.clone()
    detached = x.detach()

    assert cloned.requires_grad is True
    assert detached.requires_grad is False
    np.testing.assert_allclose(cloned.numpy(), x.numpy())
    np.testing.assert_allclose(detached.numpy(), x.numpy())

    (x * x).sum().backward()
    assert x.grad is not None
    x.zero_grad()
    assert x.grad is None


def test_no_grad_context_disables_graph_recording() -> None:
    x = ts.tensor([1.0, 2.0], requires_grad=True)

    with ts.no_grad():
        y = x * 2

    assert ts.is_grad_enabled() is True
    assert y.requires_grad is False


def test_item_and_scalar_tolist() -> None:
    x = ts.tensor(3.5)

    assert x.shape == ()
    assert x.item() == pytest.approx(3.5)
    assert x.tolist() == pytest.approx(3.5)


def test_requires_grad_rejects_integer_tensors() -> None:
    with pytest.raises(Exception, match="requires_grad"):
        ts.tensor([1, 2, 3], requires_grad=True)
