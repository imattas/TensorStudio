from __future__ import annotations

import numpy as np
import pytest
import tensorstudio as ts
from tensorstudio.errors import ShapeError


def test_squeeze_unsqueeze_match_numpy() -> None:
    values = np.arange(24, dtype=np.float64).reshape(1, 2, 1, 3, 4)
    x = ts.tensor(values)

    np.testing.assert_allclose(x.squeeze().numpy(), np.squeeze(values))
    np.testing.assert_allclose(x.squeeze(2).numpy(), np.squeeze(values, axis=2))
    np.testing.assert_allclose(x.squeeze(-3).numpy(), np.squeeze(values, axis=-3))

    y = x.squeeze().unsqueeze(0).unsqueeze(-1)
    expected = np.expand_dims(np.expand_dims(np.squeeze(values), 0), -1)
    np.testing.assert_allclose(y.numpy(), expected)
    assert y.shape == expected.shape


def test_permute_and_transpose_match_numpy() -> None:
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    x = ts.tensor(values)

    np.testing.assert_allclose(x.permute(2, 0, 1).numpy(), np.transpose(values, (2, 0, 1)))
    np.testing.assert_allclose(x.permute((1, 2, 0)).numpy(), np.transpose(values, (1, 2, 0)))
    np.testing.assert_allclose(ts.permute(x, [2, 1, 0]).numpy(), np.transpose(values, (2, 1, 0)))
    np.testing.assert_allclose(x.transpose().numpy(), np.transpose(values))
    np.testing.assert_allclose(x.transpose(0, 2).numpy(), np.swapaxes(values, 0, 2))
    np.testing.assert_allclose(ts.transpose(x, 1, 2).numpy(), np.swapaxes(values, 1, 2))
    assert x.T.shape == (4, 3, 2)


def test_view_ops_preserve_shared_storage_metadata() -> None:
    x = ts.arange(24).reshape((2, 3, 4))
    y = x.permute(2, 0, 1)
    assert y.shape == (4, 2, 3)
    assert y.strides == (1, 12, 4)
    assert not y.is_contiguous

    z = x.unsqueeze(1)
    assert z.shape == (2, 1, 3, 4)
    assert z.tolist() == np.expand_dims(x.numpy(), 1).tolist()


def test_view_op_errors_are_clear() -> None:
    x = ts.arange(24).reshape((2, 3, 4))

    with pytest.raises(ShapeError, match="axes must be unique"):
        x.permute(0, 0, 1)
    with pytest.raises(ShapeError, match="expected 3 axes"):
        x.permute(0, 1)
    with pytest.raises(ShapeError, match="out of range"):
        x.unsqueeze(5)
    with pytest.raises(ShapeError, match="cannot squeeze axis"):
        x.squeeze(0)
    with pytest.raises(ShapeError, match="exactly two axes"):
        x.transpose(0)


def test_permute_transpose_autograd() -> None:
    base = ts.arange(24, dtype="float64", requires_grad=True)
    x = base.reshape((2, 3, 4))
    weights_np = np.arange(1, 25, dtype=np.float64).reshape(4, 2, 3)
    weights = ts.tensor(weights_np)

    y = (x.permute(2, 0, 1) * weights).sum()
    y.backward()
    np.testing.assert_allclose(base.grad.numpy(), np.transpose(weights_np, (1, 2, 0)).reshape(24))

    base2 = ts.arange(24, dtype="float64", requires_grad=True)
    x2 = base2.reshape((2, 3, 4))
    w2_np = np.arange(1, 25, dtype=np.float64).reshape(4, 3, 2)
    y2 = (x2.transpose() * ts.tensor(w2_np)).sum()
    y2.backward()
    np.testing.assert_allclose(base2.grad.numpy(), np.transpose(w2_np).reshape(24))


def test_squeeze_unsqueeze_autograd() -> None:
    base = ts.arange(6, dtype="float64", requires_grad=True)
    x = base.reshape((1, 2, 1, 3))
    weights_np = np.arange(1, 7, dtype=np.float64).reshape(2, 3)
    y = (x.squeeze() * ts.tensor(weights_np)).sum()
    y.backward()
    np.testing.assert_allclose(base.grad.numpy(), weights_np.reshape(6))

    base2 = ts.arange(6, dtype="float64", requires_grad=True)
    x2 = base2.reshape((2, 3))
    weights2_np = np.arange(1, 7, dtype=np.float64).reshape(1, 2, 3, 1)
    y2 = (x2.unsqueeze(0).unsqueeze(-1) * ts.tensor(weights2_np)).sum()
    y2.backward()
    np.testing.assert_allclose(base2.grad.numpy(), weights2_np.reshape(6))
