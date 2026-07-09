from __future__ import annotations

import json

import numpy as np
import tensorstudio as ts
from tensorstudio import nn


def test_tensor_save_load_roundtrip(tmp_path) -> None:
    path = tmp_path / "tensor.tsmodel"
    x = ts.tensor([[1.0, 2.0]], requires_grad=True)

    ts.save(x, path)
    loaded = ts.load(path)

    assert loaded.requires_grad is True
    np.testing.assert_allclose(loaded.numpy(), x.numpy())


def test_model_save_load_roundtrip(tmp_path) -> None:
    path = tmp_path / "model.tsmodel"
    model = nn.Sequential(nn.Linear(1, 2), nn.Tanh(), nn.Linear(2, 1))

    ts.save(model, path)
    loaded = ts.load(path)

    assert loaded(ts.ones((3, 1))).shape == (3, 1)


def test_state_dict_and_optimizer_state_save_load_roundtrip(tmp_path) -> None:
    path = tmp_path / "state.tsmodel"
    model = nn.Linear(1, 1)
    optimizer = ts.optim.Adam(model.parameters(), lr=0.01)

    ts.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
    loaded = ts.load(path)

    assert sorted(loaded["model"]) == ["bias", "weight"]
    assert loaded["optimizer"]["lr"] == 0.01


def test_npz_tensor_roundtrip_is_non_pickle(tmp_path) -> None:
    path = tmp_path / "tensor.tsnpz"
    x = ts.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    ts.save_npz(x, path)
    loaded = ts.load_npz(path)

    assert isinstance(loaded, ts.Tensor)
    assert loaded.requires_grad is True
    np.testing.assert_allclose(loaded.numpy(), x.numpy())

    info = ts.inspect_npz(path)
    compat = ts.check_npz_compatibility(path)
    assert info["compatible"] is True
    assert info["format"] == "tensorstudio.npz"
    assert info["version"] == 1
    assert info["kind"] == "tensor"
    assert info["tensor_count"] == 1
    assert info["tensors"][0]["shape"] == [2, 2]
    assert info["tensors"][0]["array_shape"] == [2, 2]
    assert compat["compatible"] is True


def test_npz_state_dict_roundtrip_loads_into_model(tmp_path) -> None:
    path = tmp_path / "state.tsnpz"
    model = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))
    state = model.state_dict()

    ts.save_npz(state, path)
    loaded = ts.load_npz(path)

    assert isinstance(loaded, dict)
    assert sorted(loaded) == ["0.bias", "0.weight", "2.bias", "2.weight"]
    model.load_state_dict(loaded)
    np.testing.assert_allclose(model.state_dict()["0.weight"].numpy(), state["0.weight"].numpy())

    info = ts.inspect_npz(path)
    assert info["kind"] == "state_dict"
    assert info["tensor_count"] == 4
    assert info["missing_arrays"] == []


def test_npz_inspection_reports_unsupported_versions(tmp_path) -> None:
    path = tmp_path / "future.tsnpz"
    metadata = {
        "format": "tensorstudio.npz",
        "version": 999,
        "kind": "tensor",
        "tensors": [
            {
                "name": "",
                "key": "arr_0",
                "dtype": "float32",
                "shape": [1],
                "requires_grad": False,
            }
        ],
    }
    with path.open("wb") as file:
        np.savez_compressed(
            file,
            arr_0=np.array([1.0], dtype=np.float32),
            __tensorstudio_metadata__=np.array(json.dumps(metadata)),
        )

    info = ts.inspect_npz(path)
    compat = ts.check_npz_compatibility(path)

    assert info["compatible"] is False
    assert info["version"] == 999
    assert "unsupported TensorStudio npz version" in info["reason"]
    assert compat["compatible"] is False
