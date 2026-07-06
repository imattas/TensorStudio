from __future__ import annotations

import numpy as np
import tensorstudio as ts
from tensorstudio import optim


def test_sgd_updates_parameter() -> None:
    p = ts.tensor([1.0, 2.0], requires_grad=True)
    loss = (p * p).sum()
    loss.backward()

    opt = optim.SGD([p], lr=0.1)
    opt.step()

    np.testing.assert_allclose(p.numpy(), np.array([0.8, 1.6]), rtol=1e-6)


def test_sgd_momentum_weight_decay_and_state_roundtrip() -> None:
    p = ts.tensor([1.0], requires_grad=True)
    (p * p).sum().backward()
    opt = optim.SGD([p], lr=0.1, momentum=0.9, weight_decay=0.1)
    opt.step()

    state = opt.state_dict()
    clone = optim.SGD([p], lr=0.01)
    clone.load_state_dict(state)

    assert clone.lr == opt.lr
    assert clone.momentum == opt.momentum
    assert clone.weight_decay == opt.weight_decay
    assert state["velocity"][0] is not None


def test_adam_updates_parameter() -> None:
    p = ts.tensor([1.0, 2.0], requires_grad=True)
    loss = (p * p).sum()
    loss.backward()

    opt = optim.Adam([p], lr=0.1)
    before = p.numpy().copy()
    opt.step()

    assert not np.allclose(p.numpy(), before)


def test_adam_state_roundtrip_and_adamw_updates_parameter() -> None:
    p = ts.tensor([1.0, 2.0], requires_grad=True)
    (p * p).sum().backward()
    opt = optim.Adam([p], lr=0.1, weight_decay=0.01)
    opt.step()

    state = opt.state_dict()
    clone = optim.Adam([p])
    clone.load_state_dict(state)
    assert clone.lr == opt.lr
    assert clone.weight_decay == opt.weight_decay
    assert clone.t == opt.t

    q = ts.tensor([1.0, 2.0], requires_grad=True)
    (q * q).sum().backward()
    adamw = optim.AdamW([q], lr=0.1, weight_decay=0.01)
    before = q.numpy().copy()
    adamw.step()
    assert not np.allclose(q.numpy(), before)


def test_zero_grad() -> None:
    p = ts.tensor([1.0], requires_grad=True)
    (p * p).sum().backward()
    assert p.grad is not None

    opt = optim.SGD([p])
    opt.zero_grad()

    assert p.grad is None
