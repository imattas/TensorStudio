from __future__ import annotations


def test_import_and_version() -> None:
    import tensorstudio as ts

    assert ts.__version__ == "1.1.0"
    assert hasattr(ts, "Tensor")
    assert hasattr(ts, "data")
    assert hasattr(ts, "nn")
    assert hasattr(ts, "optim")
