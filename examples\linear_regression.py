from __future__ import annotations

import tensorstudio as ts
from tensorstudio import nn, optim


def main() -> None:
    x = ts.tensor([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = ts.tensor([[1.0], [3.0], [5.0], [7.0], [9.0], [11.0]])

    model = nn.Linear(1, 1)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.03)

    for epoch in range(101):
        optimizer.zero_grad()
        prediction = model(x)
        loss = loss_fn(prediction, y)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"epoch={epoch:03d} loss={loss.item():.6f}")

    print("weight =", model.weight.tolist())
    print("bias =", model.bias.tolist() if model.bias is not None else None)


if __name__ == "__main__":
    main()
