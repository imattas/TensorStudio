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
