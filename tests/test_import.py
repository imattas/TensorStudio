from __future__ import annotations


def test_import_and_version() -> None:
    import tensorstudio as ts

    assert ts.__version__ == "1.6.0"
    assert hasattr(ts, "Tensor")
    assert hasattr(ts, "data")
    assert hasattr(ts, "nn")
    assert hasattr(ts, "optim")
    assert hasattr(ts, "vision")
    assert hasattr(ts, "export_onnx")
