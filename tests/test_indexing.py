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
    with pytest.raises(Exception, match="cannot mix booleans and integers"):
        _ = x[[0, True]]
    with pytest.raises(Exception, match="could not be broadcast together"):
        _ = x[[0, 1], [0, 1, 2]]
    with pytest.raises(Exception, match="boolean mask length 1 does not match axis 0 with size 2"):
        _ = x[[True]]
    with pytest.raises(Exception, match="full-shape boolean mask shape"):
        _ = x[ts.tensor([[True, False], [False, True]], dtype="bool")]
    with pytest.raises(Exception, match="prefix boolean mask shape"):
        _ = ts.arange(24).reshape((2, 3, 4))[
            ts.tensor([[True, False], [False, True]], dtype="bool")
        ]
    with pytest.raises(Exception, match="integer tensor indices must have rank at least 1"):
        _ = x[ts.tensor(0, dtype="int64")]
    with pytest.raises(Exception, match="tensor indices must have dtype int32, int64, or bool"):
        _ = x[ts.tensor([0.0, 1.0])]
    with pytest.raises(Exception, match="at most one ellipsis"):
        _ = x[..., ...]


def test_integer_list_gather_indexing_matches_numpy() -> None:
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    x = ts.from_numpy(values)

    cases = [
        [1, 0, 1],
        (slice(None), [2, 0], slice(None, None, -1)),
        (slice(None), slice(1, None), [3, -1, 0]),
        ([], slice(None), slice(None)),
        (None, [1, 0], Ellipsis),
    ]

    for key in cases:
        actual = x[key]
        expected = values[key]
        np.testing.assert_allclose(actual.numpy(), expected)
        assert actual.shape == expected.shape


def test_boolean_list_mask_indexing_matches_numpy() -> None:
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    x = ts.from_numpy(values)

    cases = [
        [True, False],
        (slice(None), [False, True, True], slice(None, None, -1)),
        (slice(None), slice(None), [True, False, True, False]),
        [False, False],
        (None, [False, True], Ellipsis),
    ]

    for key in cases:
        actual = x[key]
        expected = values[key]
        np.testing.assert_allclose(actual.numpy(), expected)
        assert actual.shape == expected.shape


def test_tensor_index_gather_and_mask_matches_numpy() -> None:
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    x = ts.from_numpy(values)

    int64_index = ts.tensor([1, 0, 1], dtype="int64")
    int32_index = ts.tensor([2, 0], dtype="int32")
    axis0_mask = ts.tensor([True, False], dtype="bool")
    axis1_mask = ts.tensor([False, True, True], dtype="bool")
    axis2_mask = ts.tensor([True, False, True, False], dtype="bool")

    cases = [
        (int64_index, np.array([1, 0, 1], dtype=np.int64)),
        (
            (slice(None), int32_index, slice(None, None, -1)),
            (slice(None), np.array([2, 0], dtype=np.int32), slice(None, None, -1)),
        ),
        (axis0_mask, np.array([True, False])),
        (
            (slice(None), axis1_mask, slice(None, None, -1)),
            (slice(None), np.array([False, True, True]), slice(None, None, -1)),
        ),
        (
            (slice(None), slice(None), axis2_mask),
            (slice(None), slice(None), np.array([True, False, True, False])),
        ),
        ((None, axis0_mask, Ellipsis), (None, np.array([True, False]), Ellipsis)),
    ]

    for key, np_key in cases:
        actual = x[key]
        expected = values[np_key]
        np.testing.assert_allclose(actual.numpy(), expected)
        assert actual.shape == expected.shape


def test_higher_rank_single_axis_integer_tensor_indexing_matches_numpy() -> None:
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    x = ts.from_numpy(values)
    leading_index = ts.tensor([[1, 0], [0, 1]], dtype="int64")
    middle_index = ts.tensor([[2, 0], [1, 2]], dtype="int32")

    cases = [
        (leading_index, np.array([[1, 0], [0, 1]], dtype=np.int64)),
        (
            (slice(None), middle_index, slice(1, 3)),
            (slice(None), np.array([[2, 0], [1, 2]], dtype=np.int32), slice(1, 3)),
        ),
        (
            (None, leading_index, Ellipsis),
            (None, np.array([[1, 0], [0, 1]], dtype=np.int64), Ellipsis),
        ),
    ]

    for key, np_key in cases:
        actual = x[key]
        expected = values[np_key]
        np.testing.assert_allclose(actual.numpy(), expected)
        assert actual.shape == expected.shape


def test_vectorized_multi_axis_integer_indexing_matches_numpy() -> None:
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    x = ts.from_numpy(values)

    row_tensor = ts.tensor([1, 0, 1], dtype="int64")
    col_tensor = ts.tensor([2, 0, 2], dtype="int64")
    row_grid = ts.tensor([[1, 0], [0, 1]], dtype="int64")
    col_grid = ts.tensor([[2, 0], [1, 2]], dtype="int64")
    row_broadcast = ts.tensor([[1], [0]], dtype="int64")
    col_broadcast = ts.tensor([[2, 0]], dtype="int64")
    last_grid = ts.tensor([[3, 1], [0, 2]], dtype="int64")
    row_mask = ts.tensor([True, False], dtype="bool")
    col_mask = ts.tensor([False, True, True], dtype="bool")

    cases = [
        (([1, 0], [2, 1]), (np.array([1, 0]), np.array([2, 1]))),
        (([1], [2, 0]), (np.array([1]), np.array([2, 0]))),
        (
            ([True, False], [False, True, True]),
            (np.array([True, False]), np.array([False, True, True])),
        ),
        (([True, False], [1, 2]), (np.array([True, False]), np.array([1, 2]))),
        (
            (slice(None), [2, 0], [3, 1]),
            (slice(None), np.array([2, 0]), np.array([3, 1])),
        ),
        (
            (slice(None), [False, True, True], [1, 3]),
            (slice(None), np.array([False, True, True]), np.array([1, 3])),
        ),
        (
            (row_tensor, col_tensor),
            (np.array([1, 0, 1]), np.array([2, 0, 2])),
        ),
        (
            (row_mask, col_mask),
            (np.array([True, False]), np.array([False, True, True])),
        ),
        (
            (row_grid, col_grid),
            (np.array([[1, 0], [0, 1]]), np.array([[2, 0], [1, 2]])),
        ),
        ((row_broadcast, col_broadcast), (np.array([[1], [0]]), np.array([[2, 0]]))),
        (
            ([1, 0], slice(None), [3, 1]),
            (np.array([1, 0]), slice(None), np.array([3, 1])),
        ),
        (
            ([True, False], slice(None), [1, 3]),
            (np.array([True, False]), slice(None), np.array([1, 3])),
        ),
        (
            (row_grid, slice(None), last_grid),
            (np.array([[1, 0], [0, 1]]), slice(None), np.array([[3, 1], [0, 2]])),
        ),
    ]

    for key, np_key in cases:
        actual = x[key]
        expected = values[np_key]
        np.testing.assert_allclose(actual.numpy(), expected)
        assert actual.shape == expected.shape

    with pytest.raises(Exception, match="could not be broadcast together"):
        _ = x[
            ts.tensor([[1, 0], [0, 1]], dtype="int64"),
            ts.tensor([[2, 0, 1]], dtype="int64"),
        ]


def test_full_shape_boolean_tensor_mask_matches_numpy() -> None:
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    x = ts.from_numpy(values)
    mask_values = values % 3 == 0
    mask = ts.from_numpy(mask_values)

    actual = x[mask]
    expected = values[mask_values]
    np.testing.assert_allclose(actual.numpy(), expected)
    assert actual.shape == expected.shape

    tuple_actual = x[(x.greater(18),)]
    tuple_expected = values[values > 18]
    np.testing.assert_allclose(tuple_actual.numpy(), tuple_expected)
    assert tuple_actual.shape == tuple_expected.shape


def test_prefix_boolean_tensor_mask_matches_numpy() -> None:
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    x = ts.from_numpy(values)
    mask_values = np.array([[True, False, True], [False, True, False]])
    mask = ts.from_numpy(mask_values)

    actual = x[mask]
    expected = values[mask_values]
    np.testing.assert_allclose(actual.numpy(), expected)
    assert actual.shape == expected.shape


def test_partial_boolean_tensor_mask_in_tuple_matches_numpy() -> None:
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    x = ts.from_numpy(values)
    mask_values = np.array(
        [
            [True, False, True, False],
            [False, True, False, True],
            [True, False, False, False],
        ],
    )
    mask = ts.from_numpy(mask_values)

    actual = x[:, mask]
    expected = values[:, mask_values]
    np.testing.assert_allclose(actual.numpy(), expected)
    assert actual.shape == expected.shape

    values4 = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    x4 = ts.from_numpy(values4)
    actual4 = x4[:, mask, 1:4:2]
    expected4 = values4[:, mask_values, 1:4:2]
    np.testing.assert_allclose(actual4.numpy(), expected4)
    assert actual4.shape == expected4.shape

    with pytest.raises(Exception, match="partial boolean mask shape"):
        _ = x[:, ts.from_numpy(mask_values[:2])]


def test_partial_boolean_tensor_mask_mixes_with_integer_indices() -> None:
    values = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    x = ts.from_numpy(values)
    mask_values = np.array(
        [
            [True, False, False, True],
            [False, True, False, False],
            [True, False, True, False],
        ],
    )
    mask = ts.from_numpy(mask_values)
    cols = ts.tensor([0, 2, 4, 1, 3], dtype="int64")
    cols_grid = ts.tensor([[0], [2]], dtype="int64")

    actual = x[:, mask, cols]
    expected = values[:, mask_values, np.array([0, 2, 4, 1, 3])]
    np.testing.assert_allclose(actual.numpy(), expected)
    assert actual.shape == expected.shape

    actual_grid = x[:, mask, cols_grid]
    expected_grid = values[:, mask_values, np.array([[0], [2]])]
    np.testing.assert_allclose(actual_grid.numpy(), expected_grid)
    assert actual_grid.shape == expected_grid.shape


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


def test_integer_list_gather_autograd_accumulates_repeated_indices() -> None:
    base = ts.arange(6, dtype="float64", requires_grad=True)
    x = base.reshape((3, 2))
    weights = ts.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype="float64")

    loss = (x[[2, 0, 2], :] * weights).sum()
    loss.backward()

    expected = np.zeros((3, 2), dtype=np.float64)
    expected[2] += [1.0, 2.0]
    expected[0] += [3.0, 4.0]
    expected[2] += [5.0, 6.0]
    assert base.grad is not None
    np.testing.assert_allclose(base.grad.numpy(), expected.reshape(6))


def test_tensor_index_autograd_accumulates_repeated_indices() -> None:
    base = ts.arange(6, dtype="float64", requires_grad=True)
    x = base.reshape((3, 2))
    index = ts.tensor([2, 0, 2], dtype="int64")
    weights = ts.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype="float64")

    loss = (x[index, :] * weights).sum()
    loss.backward()

    expected = np.zeros((3, 2), dtype=np.float64)
    expected[2] += [1.0, 2.0]
    expected[0] += [3.0, 4.0]
    expected[2] += [5.0, 6.0]
    assert base.grad is not None
    np.testing.assert_allclose(base.grad.numpy(), expected.reshape(6))


def test_full_shape_boolean_tensor_mask_autograd_scatters_back_to_source() -> None:
    base = ts.arange(6, dtype="float64", requires_grad=True)
    x = base.reshape((2, 3))
    mask = ts.from_numpy(np.array([[True, False, True], [False, True, False]]))
    weights = ts.tensor([1.0, 2.0, 3.0], dtype="float64")

    loss = (x[mask] * weights).sum()
    loss.backward()

    expected = np.zeros((2, 3), dtype=np.float64)
    expected[[0, 0, 1], [0, 2, 1]] = [1.0, 2.0, 3.0]
    assert base.grad is not None
    np.testing.assert_allclose(base.grad.numpy(), expected.reshape(6))


def test_prefix_boolean_tensor_mask_autograd_scatters_back_to_source() -> None:
    base = ts.arange(24, dtype="float64", requires_grad=True)
    x = base.reshape((2, 3, 4))
    mask = ts.from_numpy(np.array([[True, False, True], [False, True, False]]))
    weights = ts.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ],
        dtype="float64",
    )

    loss = (x[mask] * weights).sum()
    loss.backward()

    expected = np.zeros((2, 3, 4), dtype=np.float64)
    expected[0, 0] = [1.0, 2.0, 3.0, 4.0]
    expected[0, 2] = [5.0, 6.0, 7.0, 8.0]
    expected[1, 1] = [9.0, 10.0, 11.0, 12.0]
    assert base.grad is not None
    np.testing.assert_allclose(base.grad.numpy(), expected.reshape(24))


def test_partial_boolean_tensor_mask_autograd_scatters_back_to_source() -> None:
    base = ts.arange(24, dtype="float64", requires_grad=True)
    x = base.reshape((2, 3, 4))
    mask_values = np.array(
        [
            [True, False, False, True],
            [False, True, False, False],
            [True, False, True, False],
        ],
    )
    mask = ts.from_numpy(mask_values)
    weights_values = np.arange(1, 1 + 2 * int(mask_values.sum()), dtype=np.float64).reshape(
        2,
        int(mask_values.sum()),
    )
    weights = ts.from_numpy(weights_values)

    loss = (x[:, mask] * weights).sum()
    loss.backward()

    expected = np.zeros((2, 3, 4), dtype=np.float64)
    expected[:, mask_values] = weights_values
    assert base.grad is not None
    np.testing.assert_allclose(base.grad.numpy(), expected.reshape(24))


def test_partial_boolean_tensor_mask_mixed_with_integer_autograd_scatters() -> None:
    base = ts.arange(2 * 3 * 4 * 5, dtype="float64", requires_grad=True)
    x = base.reshape((2, 3, 4, 5))
    mask_values = np.array(
        [
            [True, False, False, True],
            [False, True, False, False],
            [True, False, True, False],
        ],
    )
    cols_values = np.array([0, 2, 4, 1, 3])
    mask = ts.from_numpy(mask_values)
    cols = ts.from_numpy(cols_values.astype(np.int64))
    weights_values = np.arange(1, 1 + 2 * int(mask_values.sum()), dtype=np.float64).reshape(
        2,
        int(mask_values.sum()),
    )
    weights = ts.from_numpy(weights_values)

    loss = (x[:, mask, cols] * weights).sum()
    loss.backward()

    expected = np.zeros((2, 3, 4, 5), dtype=np.float64)
    selected = np.argwhere(mask_values)
    for batch in range(2):
        for selected_index, (row, col) in enumerate(selected):
            expected[batch, row, col, cols_values[selected_index]] += weights_values[
                batch,
                selected_index,
            ]
    assert base.grad is not None
    np.testing.assert_allclose(base.grad.numpy(), expected.reshape(2 * 3 * 4 * 5))


def test_higher_rank_single_axis_integer_tensor_indexing_autograd_accumulates() -> None:
    base = ts.arange(6, dtype="float64", requires_grad=True)
    x = base.reshape((3, 2))
    index = ts.tensor([[2, 0], [2, 1]], dtype="int64")
    weights = ts.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype="float64",
    )

    loss = (x[index, :] * weights).sum()
    loss.backward()

    expected = np.zeros((3, 2), dtype=np.float64)
    expected[2] += [1.0, 2.0]
    expected[0] += [3.0, 4.0]
    expected[2] += [5.0, 6.0]
    expected[1] += [7.0, 8.0]
    assert base.grad is not None
    np.testing.assert_allclose(base.grad.numpy(), expected.reshape(6))


def test_vectorized_multi_axis_integer_indexing_autograd_accumulates() -> None:
    base = ts.arange(24, dtype="float64", requires_grad=True)
    x = base.reshape((2, 3, 4))
    rows = ts.tensor([1, 0, 1], dtype="int64")
    cols = ts.tensor([2, 0, 2], dtype="int64")
    weights = ts.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ],
        dtype="float64",
    )

    loss = (x[rows, cols] * weights).sum()
    loss.backward()

    expected = np.zeros((2, 3, 4), dtype=np.float64)
    expected[1, 2] += [1.0, 2.0, 3.0, 4.0]
    expected[0, 0] += [5.0, 6.0, 7.0, 8.0]
    expected[1, 2] += [9.0, 10.0, 11.0, 12.0]
    assert base.grad is not None
    np.testing.assert_allclose(base.grad.numpy(), expected.reshape(24))


def test_higher_rank_vectorized_multi_axis_integer_indexing_autograd_accumulates() -> None:
    base = ts.arange(24, dtype="float64", requires_grad=True)
    x = base.reshape((2, 3, 4))
    rows = ts.tensor([[1, 0], [1, 1]], dtype="int64")
    cols = ts.tensor([[2, 0], [2, 2]], dtype="int64")
    weights = ts.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            [[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]],
        ],
        dtype="float64",
    )

    loss = (x[rows, cols] * weights).sum()
    loss.backward()

    expected = np.zeros((2, 3, 4), dtype=np.float64)
    expected[1, 2] += [1.0, 2.0, 3.0, 4.0]
    expected[0, 0] += [5.0, 6.0, 7.0, 8.0]
    expected[1, 2] += [9.0, 10.0, 11.0, 12.0]
    expected[1, 2] += [13.0, 14.0, 15.0, 16.0]
    assert base.grad is not None
    np.testing.assert_allclose(base.grad.numpy(), expected.reshape(24))


def test_non_adjacent_vectorized_integer_indexing_autograd_accumulates() -> None:
    base = ts.arange(24, dtype="float64", requires_grad=True)
    x = base.reshape((2, 3, 4))
    rows = ts.tensor([1, 0, 1], dtype="int64")
    cols = ts.tensor([2, 0, 2], dtype="int64")
    weights = ts.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype="float64",
    )

    loss = (x[rows, :, cols] * weights).sum()
    loss.backward()

    expected = np.zeros((2, 3, 4), dtype=np.float64)
    expected[1, :, 2] += [1.0, 2.0, 3.0]
    expected[0, :, 0] += [4.0, 5.0, 6.0]
    expected[1, :, 2] += [7.0, 8.0, 9.0]
    assert base.grad is not None
    np.testing.assert_allclose(base.grad.numpy(), expected.reshape(24))


def test_vectorized_boolean_and_mixed_indexing_autograd_scatters() -> None:
    base = ts.arange(24, dtype="float64", requires_grad=True)
    x = base.reshape((2, 3, 4))
    weights = ts.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ],
        dtype="float64",
    )

    loss = (x[[True, False], [False, True, True], :] * weights).sum()
    loss.backward()

    expected = np.zeros((2, 3, 4), dtype=np.float64)
    expected[0, 1] += [1.0, 2.0, 3.0, 4.0]
    expected[0, 2] += [5.0, 6.0, 7.0, 8.0]
    assert base.grad is not None
    np.testing.assert_allclose(base.grad.numpy(), expected.reshape(24))

    base2 = ts.arange(24, dtype="float64", requires_grad=True)
    x2 = base2.reshape((2, 3, 4))
    mask = ts.tensor([True, False], dtype="bool")
    cols = ts.tensor([1, 3], dtype="int64")
    weights2 = ts.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float64")

    loss2 = (x2[mask, :, cols] * weights2).sum()
    loss2.backward()

    expected2 = np.zeros((2, 3, 4), dtype=np.float64)
    expected2[0, :, 1] += [1.0, 2.0, 3.0]
    expected2[0, :, 3] += [4.0, 5.0, 6.0]
    assert base2.grad is not None
    np.testing.assert_allclose(base2.grad.numpy(), expected2.reshape(24))


def test_tensor_boolean_mask_autograd_scatters_back_to_source() -> None:
    base = ts.arange(6, dtype="float64", requires_grad=True)
    x = base.reshape((3, 2))
    mask = ts.tensor([True, False, True], dtype="bool")
    weights = ts.tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float64")

    loss = (x[mask, :] * weights).sum()
    loss.backward()

    expected = np.zeros((3, 2), dtype=np.float64)
    expected[0] = [1.0, 2.0]
    expected[2] = [3.0, 4.0]
    assert base.grad is not None
    np.testing.assert_allclose(base.grad.numpy(), expected.reshape(6))


def test_boolean_list_mask_autograd_scatters_back_to_source() -> None:
    base = ts.arange(6, dtype="float64", requires_grad=True)
    x = base.reshape((3, 2))
    weights = ts.tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float64")

    loss = (x[[True, False, True], :] * weights).sum()
    loss.backward()

    expected = np.zeros((3, 2), dtype=np.float64)
    expected[0] = [1.0, 2.0]
    expected[2] = [3.0, 4.0]
    assert base.grad is not None
    np.testing.assert_allclose(base.grad.numpy(), expected.reshape(6))
