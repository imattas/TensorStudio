from __future__ import annotations

import numpy as np
import pytest
import tensorstudio as ts


def test_tensor_creation_and_properties() -> None:
    x = ts.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    assert x.shape == (2, 2)
    assert x.ndim == 2
    assert x.size == 4
    assert x.dtype == "float32"
    assert x.requires_grad is True
    assert x.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_dtype_creation() -> None:
    assert ts.tensor([1, 2, 3]).dtype == "int64"
    assert ts.tensor([True, False]).dtype == "bool"
    assert ts.tensor([1, 2, 3], dtype="int32").dtype == "int32"
    assert ts.tensor([1.0], dtype="float64").dtype == "float64"


def test_creation_helpers() -> None:
    np.testing.assert_allclose(ts.zeros((2, 2)).numpy(), np.zeros((2, 2), dtype=np.float32))
    np.testing.assert_allclose(ts.ones((2,)).numpy(), np.ones((2,), dtype=np.float32))
    np.testing.assert_allclose(ts.full((3,), 7).numpy(), np.full((3,), 7, dtype=np.float32))
    np.testing.assert_allclose(ts.arange(4).numpy(), np.arange(4, dtype=np.float32))


def test_randn_seed_is_deterministic() -> None:
    a = ts.randn((2, 2), seed=123)
    b = ts.randn((2, 2), seed=123)

    np.testing.assert_allclose(a.numpy(), b.numpy())


def test_item_and_scalar_tolist() -> None:
    x = ts.tensor(3.5)

    assert x.shape == ()
    assert x.item() == pytest.approx(3.5)
    assert x.tolist() == pytest.approx(3.5)


def test_requires_grad_rejects_integer_tensors() -> None:
    with pytest.raises(Exception, match="requires_grad"):
        ts.tensor([1, 2, 3], requires_grad=True)
