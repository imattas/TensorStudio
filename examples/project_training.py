from __future__ import annotations

import tempfile
from pathlib import Path

import tensorstudio as ts
from tensorstudio import nn, optim
from tensorstudio.data import DataLoader, TensorDataset
from tensorstudio.project import Project, ProjectConfig, Trainer, load_state_dict, save_state_dict


def main() -> None:
    x = ts.tensor([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = ts.tensor([[1.0], [3.0], [5.0], [7.0], [9.0], [11.0]])
    loader = DataLoader(TensorDataset(x, y), batch_size=3, shuffle=True, seed=7)

    model = nn.Linear(1, 1)
    trainer = Trainer(model, optim.SGD(model.parameters(), lr=0.03), nn.MSELoss())

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "linear-project"
        project = Project(root, ProjectConfig(name="linear-regression", seed=7))
        history = trainer.fit(loader, epochs=80)

        weights_path = project.checkpoint_path("linear-regression")
        save_state_dict(model, weights_path)

        restored = nn.Linear(1, 1)
        load_state_dict(restored, weights_path)

        print("project:", project.root)
        print("final loss:", f"{history.last['loss']:.6f}")
        print("prediction:", restored(ts.tensor([[6.0]])).item())


if __name__ == "__main__":
    main()
