from __future__ import annotations

import numpy as np
import pytest
import tensorstudio as ts
from tensorstudio import nn, optim


def test_linear_forward_shape() -> None:
    ts.manual_seed(0)
    layer = nn.Linear(3, 2)
    x = ts.ones((4, 3))

    y = layer(x)

    assert y.shape == (4, 2)
    assert len(layer.parameters()) == 2


def test_conv2d_forward_shape_and_parameters() -> None:
    ts.manual_seed(0)
    layer = nn.Conv2d(1, 2, kernel_size=3, padding=1)
    x = ts.ones((4, 1, 5, 5))

    y = layer(x)

    assert y.shape == (4, 2, 5, 5)
    assert [parameter.shape for parameter in layer.parameters()] == [(2, 1, 3, 3), (2,)]
    assert "Conv2d" in repr(layer)


def test_functional_conv2d_backward_through_module() -> None:
    ts.manual_seed(0)
    layer = nn.Conv2d(1, 1, kernel_size=(2, 2), bias=True)
    x = ts.ones((1, 1, 3, 3), requires_grad=True)

    loss = layer(x).sum()
    loss.backward()

    assert x.grad is not None
    assert layer.weight.grad is not None
    assert layer.bias is not None
    assert layer.bias.grad is not None
    assert x.grad.shape == x.shape
    assert layer.weight.grad.shape == layer.weight.shape
    assert layer.bias.grad.shape == layer.bias.shape


def test_pool2d_modules_are_parameter_free_and_shape_correct() -> None:
    x = ts.arange(1, 17).reshape((1, 1, 4, 4))
    max_pool = nn.MaxPool2d(kernel_size=2)
    avg_pool = nn.AvgPool2d(kernel_size=2)

    max_out = max_pool(x)
    avg_out = avg_pool(x)

    np.testing.assert_allclose(max_out.numpy(), np.array([[[[6.0, 8.0], [14.0, 16.0]]]]))
    np.testing.assert_allclose(avg_out.numpy(), np.array([[[[3.5, 5.5], [11.5, 13.5]]]]))
    assert max_pool.parameters() == []
    assert avg_pool.parameters() == []
    assert "MaxPool2d" in repr(max_pool)
    assert "AvgPool2d" in repr(avg_pool)


def test_conv_pool_sequential_shape() -> None:
    ts.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(1, 2, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(2 * 2 * 2, 3),
    )

    y = model(ts.ones((4, 1, 4, 4)))

    assert y.shape == (4, 3)


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


def test_module_introspection_freeze_apply_and_functional_linear() -> None:
    ts.manual_seed(0)
    model = nn.Sequential(nn.Linear(2, 3), nn.LeakyReLU(0.2), nn.Linear(3, 1))

    assert [name for name, _ in model.named_children()] == ["0", "1", "2"]
    assert [name for name, _ in model.named_modules()] == ["", "0", "1", "2"]
    assert model.parameter_count() == sum(parameter.size for parameter in model.parameters())
    assert model.parameter_count(trainable_only=True) == model.parameter_count()

    visited: list[str] = []
    model.apply(lambda module: visited.append(module.__class__.__name__))
    assert visited == ["Sequential", "Linear", "LeakyReLU", "Linear"]

    model.freeze()
    assert model.parameter_count(trainable_only=True) == 0
    assert all(not parameter.requires_grad for parameter in model.parameters())
    model.unfreeze()
    assert all(parameter.requires_grad for parameter in model.parameters())

    x = ts.ones((2, 2))
    direct = nn.functional.linear(x, model[0].weight, model[0].bias)
    layered = model[0](x)
    np.testing.assert_allclose(direct.numpy(), layered.numpy())
    assert "negative_slope=0.2" in repr(model[1])


def test_identity_leaky_relu_softplus_and_new_losses() -> None:
    x = ts.tensor([-2.0, 0.0, 2.0])

    assert nn.Identity()(x) is x
    np.testing.assert_allclose(nn.LeakyReLU(0.1)(x).numpy(), np.array([-0.2, 0.0, 2.0]))
    np.testing.assert_allclose(
        nn.Softplus()(x).numpy(),
        np.log1p(np.exp(x.numpy())),
        rtol=1e-6,
    )

    logits = ts.tensor([0.0, 2.0, -2.0])
    target = ts.tensor([0.0, 1.0, 0.0])
    assert nn.BCEWithLogitsLoss()(logits, target).item() > 0.0

    prediction = ts.tensor([0.0, 2.0, 4.0])
    expected = ts.tensor([0.0, 0.0, 1.0])
    assert nn.HuberLoss(delta=1.0)(prediction, expected).item() > 0.0


def test_softmax_log_softmax_and_cross_entropy() -> None:
    logits_np = np.array([[1.0, 2.0, 3.0], [1.5, -0.5, 0.25]], dtype=np.float64)
    labels_np = np.array([2, 0], dtype=np.int64)
    logits = ts.tensor(logits_np.tolist(), dtype="float64", requires_grad=True)
    labels = ts.from_numpy(labels_np)

    shifted = logits_np - logits_np.max(axis=1, keepdims=True)
    expected_probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    expected_log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))

    np.testing.assert_allclose(nn.softmax(logits, axis=1).numpy(), expected_probs, rtol=1e-6)
    np.testing.assert_allclose(
        nn.log_softmax(logits, axis=1).numpy(),
        expected_log_probs,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        nn.softmax(logits, axis=1).sum(axis=1).numpy(),
        np.ones(2),
        rtol=1e-6,
    )

    loss = nn.CrossEntropyLoss()(logits, labels)
    expected_loss = -np.mean(expected_log_probs[np.arange(labels_np.size), labels_np])
    assert loss.item() == pytest.approx(expected_loss)

    loss.backward()
    expected_grad = expected_probs.copy()
    expected_grad[np.arange(labels_np.size), labels_np] -= 1.0
    expected_grad /= labels_np.size
    np.testing.assert_allclose(logits.grad.numpy(), expected_grad, rtol=1e-6, atol=1e-6)

    per_sample = nn.functional.cross_entropy(logits.detach(), labels, reduction="none")
    assert per_sample.shape == (2,)


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
