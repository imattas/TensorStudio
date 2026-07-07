from __future__ import annotations

import numpy as np
import pytest
import tensorstudio as ts


def test_integer_and_slice_indexing_matches_numpy() -> None:
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    x = ts.from_numpy(values)

    cases = [
        0,
        -1,
        slice(None),
        (slice(None), 1),
        (1, slice(1, None), slice(None, None, 2)),
        (slice(None), slice(None, None, -1), slice(1, 4, 2)),
        (..., 2),
        (Ellipsis, slice(None, None, -1)),
        (None, slice(None), 1, slice(None)),
        (slice(None), None, slice(None), 2),
        (),
    ]

    for key in cases:
        np.testing.assert_allclose(x[key].numpy(), values[key])
        assert x[key].shape == values[key].shape


def test_scalar_indexing_returns_zero_dim_tensor() -> None:
    values = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    x = ts.from_numpy(values)

    scalar = x[1, 2, 3]

    assert scalar.shape == ()
    assert scalar.item() == pytest.approx(values[1, 2, 3])
    assert scalar.tolist() == pytest.approx(values[1, 2, 3])


def test_empty_slice_and_negative_stride_match_numpy() -> None:
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    x = ts.from_numpy(values)

    np.testing.assert_allclose(x[:, 10:20].numpy(), values[:, 10:20])
    assert x[:, 10:20].shape == values[:, 10:20].shape
    np.testing.assert_allclose(x[::-1, ::-2].numpy(), values[::-1, ::-2])


def test_indexing_rejects_unsupported_advanced_cases() -> None:
    x = ts.arange(6).reshape((2, 3))

    with pytest.raises(Exception, match="too many indices"):
        _ = x[0, 0, 0]
    with pytest.raises(Exception, match="boolean tensor indexing"):
        _ = x[True]
    with pytest.raises(Exception, match="unsupported tensor index type"):
        _ = x[[0, 1]]
    with pytest.raises(Exception, match="at most one ellipsis"):
        _ = x[..., ...]


def test_slice_autograd_scatters_back_to_source() -> None:
    base = ts.arange(12, dtype="float64", requires_grad=True)
    x = base.reshape((3, 4))
    weights = ts.tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float64")

    loss = (x[1:, ::2] * weights).sum()
    loss.backward()

    expected = np.zeros((3, 4), dtype=np.float64)
    expected[1:, ::2] = weights.numpy()
    assert base.grad is not None
    np.testing.assert_allclose(base.grad.numpy(), expected.reshape(12))


def test_integer_index_autograd_scatters_back_to_source() -> None:
    x = ts.arange(6, dtype="float64", requires_grad=True).reshape((2, 3))

    loss = (x[-1, :] * 2.0).sum()
    loss.backward()

    expected = np.zeros((2, 3), dtype=np.float64)
    expected[-1, :] = 2.0
    assert x.grad is not None
    np.testing.assert_allclose(x.grad.numpy(), expected)
