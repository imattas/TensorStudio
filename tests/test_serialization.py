from __future__ import annotations

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


def test_npz_metadata_and_safetensors_roundtrip(tmp_path) -> None:
    state = {
        "weight": ts.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "bias": ts.tensor([0.5, -0.5]),
    }
    npz_path = tmp_path / "state.tsnpz"
    safe_path = tmp_path / "state.safetensors"

    ts.save_npz(state, npz_path, metadata={"task": "demo"})
    metadata = ts.load_npz_metadata(npz_path)
    inspected = ts.inspect_model_metadata(npz_path)
    ts.save_safetensors(state, safe_path, metadata={"task": "demo"})
    safe_loaded = ts.load_safetensors(safe_path)
    safe_metadata = ts.inspect_model_metadata(safe_path)

    assert metadata["version"] == 2
    assert metadata["metadata"]["task"] == "demo"
    assert inspected["tensor_count"] == 2
    assert sorted(safe_loaded) == ["bias", "weight"]
    np.testing.assert_allclose(safe_loaded["weight"].numpy(), state["weight"].numpy())
    assert safe_metadata["tensor_count"] == 2
    assert safe_metadata["metadata"]["task"] == "demo"
