from __future__ import annotations

import numpy as np
import pytest
import tensorstudio as ts
from tensorstudio import nn


def test_dataset_manifest_validation_and_cache(tmp_path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    sample = root / "sample.txt"
    sample.write_text("tensorstudio\n", encoding="utf-8")

    manifest = ts.data.build_dataset_manifest(root)
    assert len(manifest.entries) == 1
    assert manifest.validate()["ok"] is True
    assert ts.data.file_sha256(sample) == manifest.entries[0].sha256

    path = tmp_path / "manifest.json"
    ts.data.save_dataset_manifest(root, path)
    assert ts.data.validate_dataset_manifest(path)["ok"] is True

    sample.write_text("changed\n", encoding="utf-8")
    validation = ts.data.validate_dataset_manifest(path)
    assert validation["ok"] is False
    assert validation["changed"] == ["sample.txt"]

    dataset = ts.data.from_arrays([[1.0], [2.0]], [0, 1])
    cached = ts.data.cache_dataset(dataset, max_items=1)
    assert cached[0][0].tolist() == [1.0]
    assert cached.cache_size == 1
    assert cached[1][0].tolist() == [2.0]
    assert cached.cache_size == 1


def test_attention_and_transformer_block_shapes() -> None:
    query = ts.randn((1, 2, 3, 4), seed=1)
    value = ts.randn((1, 2, 3, 5), seed=2)
    output = nn.scaled_dot_product_attention(query, query, value, causal=True)
    assert output.shape == (1, 2, 3, 5)

    attention = nn.MultiHeadSelfAttention(embed_dim=4, num_heads=2)
    x = ts.randn((2, 3, 4), seed=3)
    assert attention(x, causal=True).shape == (2, 3, 4)

    block = nn.TransformerEncoderBlock(embed_dim=4, num_heads=2, mlp_ratio=2.0)
    assert block(x).shape == (2, 3, 4)


def test_sparse_csr_roundtrip_and_matmul() -> None:
    dense = ts.tensor([[0.0, 2.0], [3.0, 0.0]])
    csr = ts.csr_from_dense(dense)
    assert csr.nnz == 2
    np.testing.assert_allclose(csr.to_dense().numpy(), dense.numpy())
    np.testing.assert_allclose((csr @ ts.ones((2, 1))).numpy(), [[2.0], [3.0]])

    coo = csr.to_coo()
    restored = ts.csr_from_coo(coo)
    np.testing.assert_allclose(restored.to_dense().numpy(), dense.numpy())


def test_quantization_calibration_and_error() -> None:
    x = ts.tensor([-1.0, 0.0, 2.0])
    stats = ts.quantization.calibrate_tensor(x)
    assert stats.min_value == -1.0
    assert stats.max_value == 2.0
    merged = ts.quantization.merge_calibration_stats([stats, stats])
    assert merged.count == 6
    state_stats = ts.quantization.calibrate_state_dict({"x": x})
    assert state_stats["x"].absmax == 2.0
    error = ts.quantization.quantization_error(x)
    assert set(error) == {"mae", "mse", "max_abs_error"}


def test_from_dlpack_cpu_copy() -> None:
    array = np.asarray([1.0, 2.0], dtype=np.float32)
    tensor = ts.from_dlpack(array)
    np.testing.assert_allclose(tensor.numpy(), array)


def test_onnxruntime_provider_diagnostics(tmp_path) -> None:
    model = nn.Sequential(nn.Linear(2, 1))
    path = tmp_path / "linear.onnx"
    ts.export_onnx(model, path, input_shape=(1, 2))

    providers = ts.onnxruntime_available_providers()
    assert isinstance(providers, list)
    compatibility = ts.check_onnxruntime_compatibility(path)
    assert "available" in compatibility
    assert "loadable" in compatibility

    if ts.onnxruntime_is_available():
        with pytest.raises(ValueError):
            ts.interchange.OnnxRuntimeModel(path, providers=["DefinitelyMissingProvider"])
        fallback = ts.run_onnx(path, ts.ones((1, 2)), providers=["DefinitelyMissingProvider"])
        assert fallback.shape == (1, 1)
