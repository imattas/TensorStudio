from __future__ import annotations

import numpy as np
import tensorstudio as ts


def test_from_numpy_copies_data() -> None:
    array = np.array([[1.0, 2.0]], dtype=np.float32)
    tensor = ts.from_numpy(array)
    array[0, 0] = 99.0

    np.testing.assert_allclose(tensor.numpy(), np.array([[1.0, 2.0]], dtype=np.float32))


def test_numpy_returns_copy() -> None:
    tensor = ts.tensor([1.0, 2.0])
    array = tensor.numpy()
    array[0] = 99.0

    np.testing.assert_allclose(tensor.numpy(), np.array([1.0, 2.0], dtype=np.float32))


def test_numpy_dtype_roundtrip() -> None:
    for dtype in [np.float32, np.float64, np.int32, np.int64, np.bool_]:
        array = np.array([0, 1, 2], dtype=dtype)
        tensor = ts.from_numpy(array)
        assert tensor.numpy().dtype == array.dtype
        np.testing.assert_array_equal(tensor.numpy(), array)
