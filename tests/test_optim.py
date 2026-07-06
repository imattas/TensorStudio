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


def test_adam_updates_parameter() -> None:
    p = ts.tensor([1.0, 2.0], requires_grad=True)
    loss = (p * p).sum()
    loss.backward()

    opt = optim.Adam([p], lr=0.1)
    before = p.numpy().copy()
    opt.step()

    assert not np.allclose(p.numpy(), before)


def test_zero_grad() -> None:
    p = ts.tensor([1.0], requires_grad=True)
    (p * p).sum().backward()
    assert p.grad is not None

    opt = optim.SGD([p])
    opt.zero_grad()

    assert p.grad is None
