from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import tensorstudio as ts
from tensorstudio import ops

TensorBinary = Callable[[ts.Tensor, ts.Tensor], ts.Tensor]
NumpyBinary = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _broadcast_cases() -> list[tuple[np.ndarray, np.ndarray]]:
    return [
        (
            np.array(2.0, dtype=np.float64),
            np.array([[1.0, -2.0, 4.0]], dtype=np.float64),
        ),
        (
            np.array([1.0, -2.0, 4.0], dtype=np.float64),
            np.array([[0.5, 2.0, -1.0], [3.0, -4.0, 5.0]], dtype=np.float64),
        ),
        (
            np.array([[1.0, -2.0, 4.0]], dtype=np.float64),
            np.array([[0.5], [2.0]], dtype=np.float64),
        ),
        (
            np.arange(6, dtype=np.float64).reshape(2, 1, 3) - 2.0,
            np.arange(4, dtype=np.float64).reshape(1, 4, 1) + 1.0,
        ),
    ]


@pytest.mark.parametrize(
    ("name", "tensor_op", "numpy_op"),
    [
        ("add", ops.add, np.add),
        ("sub", ops.sub, np.subtract),
        ("mul", ops.mul, np.multiply),
        ("div", ops.div, np.divide),
        ("equal", ops.equal, np.equal),
        ("not_equal", ops.not_equal, np.not_equal),
        ("less", ops.less, np.less),
        ("less_equal", ops.less_equal, np.less_equal),
        ("greater", ops.greater, np.greater),
        ("greater_equal", ops.greater_equal, np.greater_equal),
    ],
)
@pytest.mark.parametrize(("left_np", "right_np"), _broadcast_cases())
def test_binary_elementwise_broadcasting_matches_numpy(
    name: str,
    tensor_op: TensorBinary,
    numpy_op: NumpyBinary,
    left_np: np.ndarray,
    right_np: np.ndarray,
) -> None:
    left = ts.from_numpy(left_np)
    right = ts.from_numpy(right_np)

    actual = tensor_op(left, right).numpy()
    expected = numpy_op(left_np, right_np)

    if expected.dtype == np.bool_:
        np.testing.assert_array_equal(actual, expected, err_msg=name)
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7, err_msg=name)


@pytest.mark.parametrize(
    ("name", "tensor_op"),
    [
        ("add", ops.add),
        ("sub", ops.sub),
        ("mul", ops.mul),
        ("div", ops.div),
        ("equal", ops.equal),
        ("not_equal", ops.not_equal),
        ("less", ops.less),
        ("less_equal", ops.less_equal),
        ("greater", ops.greater),
        ("greater_equal", ops.greater_equal),
    ],
)
def test_binary_elementwise_broadcast_errors_are_consistent(
    name: str,
    tensor_op: TensorBinary,
) -> None:
    with pytest.raises(Exception, match="broadcast"):
        tensor_op(ts.ones((2, 3)), ts.ones((4,)))


def test_operator_broadcasting_paths_match_functional_ops() -> None:
    left = ts.tensor([[1.0, 2.0, 3.0]])
    right = ts.tensor([[10.0], [20.0]])

    np.testing.assert_allclose((left + right).numpy(), ops.add(left, right).numpy())
    np.testing.assert_allclose((left - right).numpy(), ops.sub(left, right).numpy())
    np.testing.assert_allclose((left * right).numpy(), ops.mul(left, right).numpy())
    np.testing.assert_allclose((left / right).numpy(), ops.div(left, right).numpy())
    np.testing.assert_array_equal((left == right).numpy(), ops.equal(left, right).numpy())
    np.testing.assert_array_equal((left != right).numpy(), ops.not_equal(left, right).numpy())
    np.testing.assert_array_equal((left < right).numpy(), ops.less(left, right).numpy())
    np.testing.assert_array_equal((left <= right).numpy(), ops.less_equal(left, right).numpy())
    np.testing.assert_array_equal((left > right).numpy(), ops.greater(left, right).numpy())
    np.testing.assert_array_equal((left >= right).numpy(), ops.greater_equal(left, right).numpy())
