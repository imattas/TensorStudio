from __future__ import annotations

import numpy as np
import pytest
import tensorstudio as ts
from tensorstudio import nn

onnx = pytest.importorskip("onnx")


def test_export_onnx_sequential_conv_classifier(tmp_path) -> None:
    ts.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(1, 2, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(2 * 2 * 2, 3),
    )
    path = tmp_path / "classifier.onnx"

    exported = ts.export_onnx(model, path, input_shape=(1, 1, 4, 4))
    loaded = onnx.load(exported)
    onnx.checker.check_model(loaded)

    assert path.exists()
    assert [node.op_type for node in loaded.graph.node] == [
        "Conv",
        "Relu",
        "MaxPool",
        "Flatten",
        "Gemm",
    ]


def test_export_onnx_vision_classifier(tmp_path) -> None:
    model = ts.vision.TinyConvClassifier((1, 4, 4), num_classes=2, hidden_channels=2)
    path = tmp_path / "vision.onnx"

    ts.export_onnx(model, path, input_shape=(1, 1, 4, 4))
    loaded = onnx.load(path)

    assert loaded.graph.output[0].name == "output"
    assert [dim.dim_value for dim in loaded.graph.output[0].type.tensor_type.shape.dim] == [1, 2]


def test_export_onnx_configurable_image_classifier(tmp_path) -> None:
    model = ts.vision.ImageClassifier((1, 8, 8), num_classes=3, channels=(2, 4))
    path = tmp_path / "image_classifier.onnx"

    ts.export_onnx(model, path, input_shape=(1, 1, 8, 8))
    loaded = onnx.load(path)

    assert [node.op_type for node in loaded.graph.node] == [
        "Conv",
        "Relu",
        "MaxPool",
        "Conv",
        "Relu",
        "MaxPool",
        "Flatten",
        "Gemm",
    ]


def test_onnx_metadata_inspection_and_runtime_diagnostics(tmp_path) -> None:
    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))
    path = tmp_path / "mlp.onnx"

    ts.export_onnx(model, path, input_shape=(4, 2), input_name="features", output_name="score")

    info = ts.inspect_onnx(path)
    assert info["check_passed"] is True
    assert info["graph_name"] == "TensorStudioModel"
    assert info["inputs"] == [{"name": "features", "dtype": "float", "shape": [4, 2]}]
    assert info["outputs"] == [{"name": "score", "dtype": "float", "shape": [4, 1]}]
    assert info["node_count"] == 3
    assert info["op_counts"] == {"Gemm": 2, "Relu": 1}
    assert "" in info["opsets"]

    runtime = ts.onnx_runtime_info(providers=["DefinitelyMissingProvider"])
    assert runtime["requested_providers"] == ["DefinitelyMissingProvider"]
    if runtime["available"]:
        assert "DefinitelyMissingProvider" in runtime["unsupported_providers"]
    else:
        assert runtime["selected_providers"] == []
        assert "tensorstudio[onnxruntime]" in runtime["reason"]

    compatibility = ts.check_onnx_runtime_compatibility(
        path,
        providers=["DefinitelyMissingProvider"],
    )
    assert compatibility["onnx_check_passed"] is True
    assert compatibility["requested_providers"] == ["DefinitelyMissingProvider"]
    if compatibility["runtime_available"]:
        assert compatibility["compatible"] is False
        assert "DefinitelyMissingProvider" in compatibility["unsupported_providers"]
    else:
        assert compatibility["compatible"] is None


def test_run_onnx_inference_matches_tensorstudio_forward(tmp_path) -> None:
    ort = pytest.importorskip("onnxruntime")
    if "CPUExecutionProvider" not in ort.get_available_providers():
        pytest.skip("ONNX Runtime CPUExecutionProvider is not available")

    ts.manual_seed(7)
    model = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))
    x = ts.tensor([[0.0, 1.0], [2.0, -1.0], [0.5, 0.25]], dtype="float32")
    path = tmp_path / "runtime.onnx"

    ts.export_onnx(model, path, input_shape=x.shape, input_name="features", output_name="score")
    expected = model(x).numpy()

    tensor_outputs = ts.run_onnx_inference(
        path,
        x,
        providers=["CPUExecutionProvider"],
    )
    array_outputs = ts.run_onnx_inference(
        path,
        {"features": x.numpy()},
        providers=["CPUExecutionProvider"],
        output_names="score",
        as_tensor=False,
    )

    assert list(tensor_outputs) == ["score"]
    assert isinstance(tensor_outputs["score"], ts.Tensor)
    assert isinstance(array_outputs["score"], np.ndarray)
    np.testing.assert_allclose(tensor_outputs["score"].numpy(), expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(array_outputs["score"], expected, rtol=1e-5, atol=1e-6)

    with pytest.raises(ValueError, match="unknown ONNX input"):
        ts.run_onnx_inference(
            path,
            {"features": x, "extra": x},
            providers=["CPUExecutionProvider"],
        )
    with pytest.raises(ValueError, match="unsupported ONNX Runtime provider"):
        ts.run_onnx_inference(path, x, providers=["DefinitelyMissingProvider"])
