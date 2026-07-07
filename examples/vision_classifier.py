from __future__ import annotations

import numpy as np
import tensorstudio as ts
from tensorstudio import nn, optim


def main() -> None:
    ts.manual_seed(0)
    image = np.zeros((8, 8, 1), dtype=np.uint8)
    image[2:6, 2:6, 0] = 255
    x = ts.vision.to_tensor(image).reshape((1, 1, 8, 8))
    target = ts.tensor([1], dtype="int64")

    model = ts.vision.TinyConvClassifier((1, 8, 8), num_classes=2, hidden_channels=4)
    optimizer = optim.SGD(model.parameters(), lr=0.05)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(5):
        optimizer.zero_grad()
        loss = loss_fn(model(x), target)
        loss.backward()
        optimizer.step()

    print("vision logits:", model(x).tolist())
    print("vision loss:", loss.item())


if __name__ == "__main__":
    main()
