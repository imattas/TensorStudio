from __future__ import annotations

import tensorstudio as ts
from tensorstudio import nn, optim


def main() -> None:
    x = ts.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = ts.tensor([[0.0], [1.0], [1.0], [0.0]])

    model = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1), nn.Sigmoid())
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    for epoch in range(201):
        optimizer.zero_grad()
        prediction = model(x)
        loss = loss_fn(prediction, y)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"epoch={epoch:03d} loss={loss.item():.6f}")

    print("predictions =", model(x).tolist())


if __name__ == "__main__":
    main()
