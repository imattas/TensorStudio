from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import tensorstudio as ts
from tensorstudio import nn


def main() -> None:
    model = nn.Sequential(
        nn.Conv2d(1, 2, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(2 * 2 * 2, 3),
    )
    with TemporaryDirectory() as directory:
        path = Path(directory) / "classifier.onnx"
        try:
            ts.export_onnx(model, path, input_shape=(1, 1, 4, 4))
        except ImportError as exc:
            print(exc)
            return
        print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
