from __future__ import annotations

import tensorstudio as ts
from tensorstudio import nn, optim


def test_linear_forward_shape() -> None:
    layer = nn.Linear(3, 2)
    x = ts.ones((4, 3))

    y = layer(x)

    assert y.shape == (4, 2)
    assert len(layer.parameters()) == 2


def test_tiny_training_loop_reduces_loss() -> None:
    model = nn.Linear(1, 1)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05)
    x = ts.tensor([[0.0], [1.0], [2.0], [3.0]])
    y = ts.tensor([[1.0], [3.0], [5.0], [7.0]])

    first_loss = loss_fn(model(x), y).item()
    for _ in range(80):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    final_loss = loss_fn(model(x), y).item()

    assert final_loss < first_loss


def test_sequential_and_eval_train() -> None:
    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))

    assert model.eval().training is False
    assert all(module.training is False for module in model)
    assert model.train().training is True
    assert model(ts.ones((2, 2))).shape == (2, 1)
