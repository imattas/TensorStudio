from __future__ import annotations

from pathlib import Path

import tensorstudio as ts
from tensorstudio import nn


def main() -> None:
    path = Path("model.tsmodel")
    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))
    x = ts.ones((2, 2))

    ts.save(model, path)
    loaded = ts.load(path)

    print("saved to", path)
    print("output =", loaded(x).tolist())


if __name__ == "__main__":
    main()
