from __future__ import annotations

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
