from __future__ import annotations

import numpy as np
import tensorstudio as ts
from tensorstudio import nn, optim
from tensorstudio.data import DataLoader, TensorDataset
from tensorstudio.project import (
    Project,
    ProjectConfig,
    Trainer,
    load_checkpoint,
    load_state_dict,
    save_checkpoint,
    save_state_dict,
)


def test_project_config_and_workspace_roundtrip(tmp_path) -> None:
    config = ProjectConfig(
        name="demo",
        version="1.0.0",
        seed=123,
        metadata={"task": "regression", "debug": True},
    )
    project = Project(tmp_path / "demo-project", config=config)

    assert project.config_path.exists()
    assert project.checkpoints_dir.exists()
    assert project.artifacts_dir.exists()
    assert project.logs_dir.exists()
    assert project.checkpoint_path("weights").name == "weights.tsnpz"

    loaded = Project.load(project.root)
    assert loaded.config.name == "demo"
    assert loaded.config.seed == 123
    assert loaded.config.metadata["task"] == "regression"


def test_trainer_fit_reduces_loss_and_evaluate_runs() -> None:
    x = ts.tensor([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = ts.tensor([[1.0], [3.0], [5.0], [7.0], [9.0], [11.0]])
    loader = DataLoader(TensorDataset(x, y), batch_size=3, shuffle=False)

    model = nn.Linear(1, 1)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.03)
    trainer = Trainer(model, optimizer, loss_fn)

    initial = loss_fn(model(x), y).item()
    history = trainer.fit(loader, epochs=80)
    final = loss_fn(model(x), y).item()
    metrics = trainer.evaluate(loader)

    assert history.last["epoch"] == 80.0
    assert history.last["loss"] < initial
    assert final < initial * 0.05
    assert metrics["loss"] >= 0.0


def test_project_state_dict_and_full_checkpoint_roundtrip(tmp_path) -> None:
    x = ts.tensor([[0.0], [1.0], [2.0]])
    model = nn.Linear(1, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    prediction = model(x).numpy()

    state_path = tmp_path / "weights.tsnpz"
    save_state_dict(model, state_path)
    restored = nn.Linear(1, 1)
    load_state_dict(restored, state_path)
    np.testing.assert_allclose(restored(x).numpy(), prediction, rtol=1e-6, atol=1e-6)

    checkpoint_path = tmp_path / "trusted.tsmodel"
    save_checkpoint(model, checkpoint_path, optimizer=optimizer, metadata={"epoch": 4})
    restored_optimizer = optim.Adam(restored.parameters(), lr=0.5)
    checkpoint = load_checkpoint(restored, checkpoint_path, optimizer=restored_optimizer)

    assert checkpoint["metadata"]["epoch"] == 4
    assert checkpoint["format"] == "tensorstudio.checkpoint"
    assert checkpoint["version"] == 2
    with np.testing.assert_raises(ValueError):
        ts.inspect_model_metadata(checkpoint_path)
    metadata = ts.inspect_model_metadata(checkpoint_path, trusted_pickle=True)
    assert metadata["has_optimizer"] is True
    assert metadata["model_tensor_count"] == 2
    np.testing.assert_allclose(restored(x).numpy(), prediction, rtol=1e-6, atol=1e-6)
