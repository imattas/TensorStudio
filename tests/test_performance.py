from __future__ import annotations

import numpy as np
import tensorstudio as ts


def test_performance_info_shape() -> None:
    info = ts.performance_info()

    assert isinstance(info["num_threads"], int)
    assert info["num_threads"] >= 1
    assert isinstance(info["threads_enabled"], bool)
    assert isinstance(info["storage_pool_enabled"], bool)
    assert isinstance(info["blas_enabled"], bool)
    assert isinstance(info["simd_enabled"], bool)
    assert isinstance(info["simd_level"], str)


def test_threaded_fast_paths_match_numpy() -> None:
    previous_threads = ts.get_num_threads()
    ts.set_num_threads(2)
    try:
        left = np.linspace(-1.0, 1.0, 65536, dtype=np.float32)
        right = np.linspace(0.5, 2.0, 65536, dtype=np.float32)
        x = ts.from_numpy(left)
        y = ts.from_numpy(right)

        np.testing.assert_allclose((x + y).numpy(), left + right, rtol=1e-6)
        np.testing.assert_allclose((x * y).numpy(), left * right, rtol=1e-6)
        np.testing.assert_allclose(x.relu().numpy(), np.maximum(left, 0.0), rtol=1e-6)
        np.testing.assert_allclose(x.sum().item(), np.sum(left, dtype=np.float64), atol=1e-5)
    finally:
        ts.set_num_threads(previous_threads)


def test_matmul_fast_path_matches_numpy_float64() -> None:
    rng = np.random.default_rng(123)
    left = rng.standard_normal((32, 48)).astype(np.float64)
    right = rng.standard_normal((48, 16)).astype(np.float64)

    actual = ts.from_numpy(left) @ ts.from_numpy(right)

    np.testing.assert_allclose(actual.numpy(), left @ right, rtol=1e-10, atol=1e-10)
