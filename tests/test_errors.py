from __future__ import annotations

import pytest
import tensorstudio as ts
from tensorstudio.errors import DTypeError, ShapeError


def test_shape_errors_include_actionable_context() -> None:
    with pytest.raises(
        ShapeError,
        match=r"cannot broadcast shapes \(2, 3\) and \(4, 3\).*axis -2.*2 and 4",
    ):
        _ = ts.ones((2, 3)) + ts.ones((4, 3))

    with pytest.raises(
        ShapeError,
        match=r"cannot reshape tensor with 6 elements.*requested shape has 4 elements",
    ):
        ts.arange(6).reshape((4,))

    with pytest.raises(ShapeError, match=r"shape entries must be integers.*str value 'bad'"):
        ts.zeros(("bad",))


def test_dtype_errors_include_supported_values_and_actual_dtype() -> None:
    with pytest.raises(DTypeError, match=r"unsupported dtype 'complex64'.*supported dtypes"):
        ts.zeros((1,), dtype="complex64")

    with pytest.raises(DTypeError, match=r"dtype must be a string or None, got int"):
        ts.tensor([1.0], dtype=123)

    with pytest.raises(DTypeError, match=r"requires_grad.*floating point tensors.*int64"):
        ts.tensor([1, 2, 3], requires_grad=True)

    with pytest.raises(
        DTypeError,
        match=r"concat.*tensor 0 has dtype float32.*tensor 1 has dtype float64",
    ):
        ts.concat([ts.ones((1,), dtype="float32"), ts.ones((1,), dtype="float64")])


def test_indexing_errors_include_shape_rank_and_bad_type() -> None:
    x = ts.arange(6).reshape((2, 3))

    with pytest.raises(
        ShapeError,
        match=r"too many indices: got 3 indexing entries for rank 2 tensor",
    ):
        _ = x[0, 0, 0]

    with pytest.raises(ShapeError, match=r"boolean scalar indexing.*shape \(2, 3\)"):
        _ = x[True]

    with pytest.raises(ShapeError, match=r"unsupported tensor index type list value \[0, 1\]"):
        _ = x[[0, 1]]

    with pytest.raises(ShapeError, match=r"at most one ellipsis.*shape \(2, 3\)"):
        _ = x[..., ...]

    with pytest.raises(ShapeError, match=r"invalid slice slice\(0, 2, 0\).*axis 0.*size 2"):
        _ = x[slice(0, 2, 0)]
