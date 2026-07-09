from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
import tensorstudio as ts
from tensorstudio import nn

HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def test_inspect_keras_archive_metadata(tmp_path) -> None:
    archive_path = tmp_path / "model.keras"
    metadata = {"keras_version": "3.4.0", "date_saved": "2026-07-08@00:00:00"}
    config = {
        "class_name": "Sequential",
        "registered_name": "Sequential",
        "config": {
            "layers": [
                {"class_name": "InputLayer"},
                {"class_name": "Dense"},
            ],
        },
    }
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("metadata.json", json.dumps(metadata))
        archive.writestr("config.json", json.dumps(config))
        archive.writestr("model.weights.h5", HDF5_SIGNATURE)

    info = ts.inspect_keras(archive_path)
    routed = ts.inspect_model_format(archive_path)

    assert info["format"] == "keras"
    assert info["safe_metadata_only"] is True
    assert info["has_metadata_json"] is True
    assert info["has_config_json"] is True
    assert info["has_weights"] is True
    assert info["keras_version"] == "3.4.0"
    assert info["model_class"] == "Sequential"
    assert info["layer_count"] == 2
    assert info["layer_classes"] == ["InputLayer", "Dense"]
    assert routed["format"] == "keras"


def test_inspect_saved_model_directory_metadata(tmp_path) -> None:
    root = tmp_path / "saved_model"
    variables = root / "variables"
    assets = root / "assets"
    variables.mkdir(parents=True)
    assets.mkdir()
    (root / "saved_model.pb").write_bytes(b"not-a-real-protobuf")
    (variables / "variables.index").write_bytes(b"index")
    (variables / "variables.data-00000-of-00001").write_bytes(b"data")
    (assets / "labels.txt").write_text("cat\n", encoding="utf-8")

    info = ts.inspect_saved_model(root)
    routed = ts.inspect_model_format(root)

    assert info["format"] == "tensorflow_saved_model"
    assert info["safe_metadata_only"] is True
    assert info["has_saved_model_pb"] is True
    assert info["has_variables_dir"] is True
    assert sorted(info["variables"]) == [
        str(Path("variables") / "variables.data-00000-of-00001"),
        str(Path("variables") / "variables.index"),
    ]
    assert info["assets"] == [str(Path("assets") / "labels.txt")]
    assert info["loadable_by_tensorstudio"] is False
    assert routed["format"] == "tensorflow_saved_model"


def test_inspect_hdf5_signature_without_model_loading(tmp_path) -> None:
    path = tmp_path / "weights.h5"
    path.write_bytes(HDF5_SIGNATURE + b"minimal placeholder")

    info = ts.inspect_hdf5(path)
    routed = ts.inspect_model_format(path)

    assert info["format"] == "hdf5"
    assert info["safe_metadata_only"] is True
    assert info["has_hdf5_signature"] is True
    assert info["loadable_by_tensorstudio"] is False
    assert routed["format"] == "hdf5"


def test_inspect_tflite_identifier_without_model_loading(tmp_path) -> None:
    path = tmp_path / "model.tflite"
    path.write_bytes(b"\x14\x00\x00\x00TFL3minimal-flatbuffer-placeholder")

    info = ts.inspect_tflite(path)
    routed = ts.inspect_model_format(path)

    assert info["format"] == "tensorflow_lite"
    assert info["container"] == "flatbuffer"
    assert info["safe_metadata_only"] is True
    assert info["has_tflite_identifier"] is True
    assert info["loadable_by_tensorstudio"] is False
    assert routed["format"] == "tensorflow_lite"


def test_inspect_model_format_routes_onnx(tmp_path) -> None:
    pytest.importorskip("onnx")
    model = nn.Sequential(nn.Linear(2, 1))
    path = tmp_path / "model.onnx"

    ts.export_onnx(model, path, input_shape=(3, 2))
    info = ts.inspect_model_format(path)

    assert info["format"] == "onnx"
    assert info["safe_metadata_only"] is True
    assert info["check_passed"] is True
    assert info["op_counts"] == {"Gemm": 1}


def test_inspect_model_format_rejects_unknown_files(tmp_path) -> None:
    path = tmp_path / "model.txt"
    path.write_text("not a model", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported model format"):
        ts.inspect_model_format(path)
