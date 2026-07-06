from __future__ import annotations

import tensorstudio as ts
from tensorstudio import nn, optim


def test_linear_forward_shape() -> None:
    ts.manual_seed(0)
    layer = nn.Linear(3, 2)
    x = ts.ones((4, 3))

    y = layer(x)

    assert y.shape == (4, 2)
    assert len(layer.parameters()) == 2


def test_tiny_training_loop_reduces_loss() -> None:
    ts.manual_seed(0)
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
    ts.manual_seed(0)
    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))

    assert model.eval().training is False
    assert all(module.training is False for module in model)
    assert all(module.training is False for module in model.modules())
    assert model.train().training is True
    assert model(ts.ones((2, 2))).shape == (2, 1)


def test_named_parameters_state_dict_and_repr() -> None:
    ts.manual_seed(0)
    model = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))

    names = [name for name, _ in model.named_parameters()]
    assert names == ["0.weight", "0.bias", "2.weight", "2.bias"]
    assert len(model.modules()) == 4
    assert "Sequential" in repr(model)

    state = model.state_dict()
    model[0].weight._assign(ts.zeros(model[0].weight.shape))
    model.load_state_dict(state)

    assert model[0].weight.tolist() == state["0.weight"].tolist()


def test_dropout_flatten_and_losses() -> None:
    x = ts.ones((2, 3))
    dropout = nn.Dropout(p=0.5, seed=123)
    dropout.train()
    dropped = dropout(x)

    assert dropped.shape == x.shape
    assert set(dropped.numpy().ravel().tolist()).issubset({0.0, 2.0})
    assert dropout.eval()(x).tolist() == x.tolist()

    flattened = nn.Flatten()(ts.ones((2, 3, 4)))
    assert flattened.shape == (2, 12)

    prediction = ts.tensor([0.25, 0.75])
    target = ts.tensor([0.0, 1.0])
    assert nn.L1Loss()(prediction, target).item() > 0.0
    assert nn.BCELoss()(prediction, target).item() > 0.0
