# Training Loops

TensorStudio training loops are explicit Python loops using eager tensors,
modules, losses, and optimizers.

## Linear Regression

```python
import tensorstudio as ts
from tensorstudio import nn, optim

x = ts.tensor([[0.0], [1.0], [2.0], [3.0]])
y = ts.tensor([[1.0], [3.0], [5.0], [7.0]])

model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)

for step in range(100):
    optimizer.zero_grad()
    prediction = model(x)
    loss = loss_fn(prediction, y)
    loss.backward()
    optimizer.step()
```

## Optimizer Contract

Optimizers expect an iterable of tensors that require gradients.

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

Every optimizer supports:

- `zero_grad()`
- `step()`

## Gradient Accumulation

Gradients accumulate by default. This is useful for some algorithms, but most
training loops clear gradients each step:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## Determinism

Use seeded tensor creation in examples and tests:

```python
w = ts.randn((4, 4), seed=123)
```

## Practical Tips

- Keep tensors modest in `1.1.0`; kernels are simple CPU loops.
- Prefer `float32` unless you need `float64`.
- Compare with NumPy in tests for expected numerical values.
- Watch shape errors carefully around broadcasting and matrix multiplication.
