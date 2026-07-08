from __future__ import annotations

import json

import numpy as np
import pytest
import tensorstudio as ts
import tensorstudio.interchange.onnx as onnx_module
from tensorstudio import nn


def test_sparse_coo_dense_roundtrip_and_matmul() -> None:
    sparse = ts.sparse_coo_tensor(
        [[0, 1], [1, 0], [1, 0]],
        [2.0, 3.0, 4.0],
        (2, 2),
        coalesced=False,
    )

    coalesced = sparse.coalesce()
    np.testing.assert_allclose(coalesced.to_dense().numpy(), [[0.0, 2.0], [7.0, 0.0]])
    np.testing.assert_allclose((coalesced @ ts.ones((2, 1))).numpy(), [[2.0], [7.0]])
    assert ts.sparse_from_dense(coalesced.to_dense()).nnz == 2


def test_public_format_datasets(tmp_path) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x,y,label\n1,2,0\n3,4,1\n", encoding="utf-8")
    csv_dataset = ts.data.from_csv(csv_path, target_column="label", target_dtype="int64")
    features, label = csv_dataset[1]
    assert features.tolist() == [3.0, 4.0]
    assert label.item() == 1

    jsonl_path = tmp_path / "data.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps({"features": [1.0, 0.0], "label": 0}),
                json.dumps({"features": [0.0, 1.0], "label": 1}),
            ]
        ),
        encoding="utf-8",
    )
    jsonl_dataset = ts.data.from_jsonl(jsonl_path, target_dtype="int64")
    assert len(jsonl_dataset) == 2
    assert jsonl_dataset[0][0].tolist() == [1.0, 0.0]

    text_path = tmp_path / "lines.txt"
    text_path.write_text("alpha\n\nbeta\n", encoding="utf-8")
    assert ts.data.from_text_lines(text_path).lines == ["alpha", "beta"]

    libsvm_path = tmp_path / "data.libsvm"
    libsvm_path.write_text("1 1:0.5 3:1.5\n0 2:2.0\n", encoding="utf-8")
    libsvm_dataset = ts.data.from_libsvm(libsvm_path, num_features=3)
    assert libsvm_dataset[0][0].tolist() == [0.5, 0.0, 1.5]


def test_language_model_helpers() -> None:
    vocab = nn.Vocabulary.build(["hello small world", "hello tensor world"])
    ids = vocab.encode("hello missing world")
    assert len(ids) == 3
    assert vocab.decode(ids).split()[0] == "hello"

    inputs, targets = nn.make_causal_lm_batch([1, 2, 3, 4], sequence_length=2)
    assert inputs.shape == (2, 2)
    assert targets.shape == (2, 2)

    model = nn.CausalLanguageModel(vocab_size=8, embedding_dim=4, max_length=4)
    logits = model(inputs)
    assert logits.shape == (2, 2, 8)
    loss = nn.causal_language_model_loss(logits, targets)
    assert loss.item() > 0


def test_quantization_roundtrip_and_report() -> None:
    x = ts.tensor([-1.0, 0.0, 1.0])
    quantized = ts.quantization.quantize_tensor(
        x,
        ts.quantization.QuantizationConfig(num_bits=8, symmetric=True),
    )

    np.testing.assert_allclose(quantized.dequantize().numpy(), x.numpy(), atol=0.02)
    fake = ts.quantization.fake_quantize(x)
    assert fake.shape == x.shape
    report = ts.quantization.quantization_report({"x": x})
    assert report["total_parameters"] == 3


def test_kernel_registry_and_model_zoo() -> None:
    ts.register_kernel("double_for_test", lambda x: x * 2.0, overwrite=True)
    try:
        output = ts.call_kernel("double_for_test", ts.ones((2,)))
        assert output.tolist() == [2.0, 2.0]
        assert ts.kernel_info("double_for_test")["backend"] == "python"
    finally:
        ts.unregister_kernel("double_for_test")

    assert "tiny_mlp" in ts.list_models()
    model = ts.create_model("tiny_mlp", input_dim=2, hidden_dim=4, output_dim=1)
    assert model(ts.ones((1, 2))).shape == (1, 1)
    assert ts.model_info("tiny_cnn")["task"] == "vision"


def test_distributed_single_process_helpers() -> None:
    config = ts.distributed.DistributedConfig()
    x = ts.tensor([1.0, 2.0])
    assert ts.distributed.all_reduce_sum(x, config).tolist() == [1.0, 2.0]
    plan = ts.distributed.data_parallel_plan(10, 4, config)
    assert plan["global_batch_size"] == 4
    assert plan["rank_batches"] == 3

    with pytest.raises(NotImplementedError):
        ts.distributed.all_reduce_sum(x, ts.distributed.DistributedConfig(world_size=2, rank=0))


def test_onnx_runtime_adapter_falls_back_to_tensorstudio_import(tmp_path) -> None:
    model = nn.Sequential(nn.Linear(2, 1))
    path = tmp_path / "linear.onnx"
    ts.export_onnx(model, path, input_shape=(1, 2))

    x = ts.ones((1, 2))
    expected = ts.import_onnx(path)(x)
    actual = ts.run_onnx(path, x, prefer_onnxruntime=False)

    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-6, atol=1e-6)
    assert isinstance(ts.onnxruntime_is_available(), bool)


def test_onnx_runtime_adapter_falls_back_when_runtime_session_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = nn.Sequential(nn.Linear(2, 1))
    path = tmp_path / "linear.onnx"
    ts.export_onnx(model, path, input_shape=(1, 2))
    x = ts.ones((1, 2))
    expected = ts.import_onnx(path)(x)

    class BrokenRuntime:
        def __init__(self, *_args, **_kwargs) -> None:
            raise RuntimeError("runtime cannot load this IR version")

    monkeypatch.setattr(onnx_module, "onnxruntime_is_available", lambda: True)
    monkeypatch.setattr(onnx_module, "OnnxRuntimeModel", BrokenRuntime)

    actual = ts.run_onnx(path, x)
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-6, atol=1e-6)
