"""Project template generation."""

from __future__ import annotations

from pathlib import Path

from tensorstudio.typing import PathLikeStr

_TEMPLATES = {
    "regression": {
        "README.md": "# TensorStudio Regression Project\n\nRun `python train.py`.\n",
        "train.py": (
            "import tensorstudio as ts\n"
            "from tensorstudio import nn, optim\n"
            "from tensorstudio.data import DataLoader, from_arrays\n"
            "from tensorstudio.project import Trainer, seed_everything\n\n"
            "seed_everything(7)\n"
            "dataset = from_arrays([[0.0], [1.0], [2.0]], [[1.0], [3.0], [5.0]])\n"
            "loader = DataLoader(dataset, batch_size=3)\n"
            "model = nn.Linear(1, 1)\n"
            "trainer = Trainer(model, optim.SGD(model.parameters(), lr=0.05), nn.MSELoss())\n"
            "print(trainer.fit(loader, epochs=10).last)\n"
        ),
    },
    "classification": {
        "README.md": "# TensorStudio Classification Project\n\nRun `python train.py`.\n",
        "train.py": (
            "import tensorstudio as ts\n"
            "from tensorstudio import nn, optim\n"
            "from tensorstudio.data import DataLoader, from_arrays\n"
            "from tensorstudio.project import Trainer, seed_everything\n\n"
            "seed_everything(7)\n"
            "dataset = from_arrays([[1.0, 0.0], [0.0, 1.0]], [0, 1])\n"
            "loader = DataLoader(dataset, batch_size=2)\n"
            "model = nn.Linear(2, 2)\n"
            "trainer = Trainer(\n"
            "    model, optim.SGD(model.parameters(), lr=0.05), nn.CrossEntropyLoss()\n"
            ")\n"
            "print(trainer.fit(loader, epochs=10).last)\n"
        ),
    },
    "vision": {
        "README.md": (
            "# TensorStudio Vision Project\n\nPlace images under `data/class_name/*.png`.\n"
        ),
        "train.py": (
            "from tensorstudio import nn, optim, vision\n"
            "from tensorstudio.data import DataLoader, from_image_folder\n"
            "from tensorstudio.project import Trainer, seed_everything\n\n"
            "seed_everything(7)\n"
            "dataset = from_image_folder('data')\n"
            "loader = DataLoader(dataset, batch_size=4, shuffle=True, seed=7)\n"
            "model = vision.ImageClassifier((3, 32, 32), num_classes=len(dataset.classes))\n"
            "trainer = Trainer(\n"
            "    model, optim.SGD(model.parameters(), lr=0.01), nn.CrossEntropyLoss()\n"
            ")\n"
            "print(trainer.fit(loader, epochs=5).last)\n"
        ),
    },
}


def create_project_template(kind: str, root: PathLikeStr, *, overwrite: bool = False) -> Path:
    """Create a small runnable project template."""

    if kind not in _TEMPLATES:
        raise ValueError(f"unknown template kind {kind!r}; choose from {sorted(_TEMPLATES)}")
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    for relative, content in _TEMPLATES[kind].items():
        path = root_path / relative
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing file: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root_path


__all__ = ["create_project_template"]
