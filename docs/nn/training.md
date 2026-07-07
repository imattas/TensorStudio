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

## Project Trainer

For repeatable small projects, `tensorstudio.project.Trainer` wraps the same
eager training steps and records epoch metrics:

```python
from tensorstudio.data import DataLoader, TensorDataset
from tensorstudio.project import Project, ProjectConfig, Trainer, save_state_dict

loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=True, seed=7)
project = Project("runs/linear", ProjectConfig(name="linear-regression", seed=7))
trainer = Trainer(model, optimizer, loss_fn)

history = trainer.fit(loader, epochs=50)
save_state_dict(model, project.checkpoint_path("weights"))

print(history.last)
```

Use this when you want a project folder, a JSON config, reusable trainer calls,
and checkpoint helpers while keeping the model code ordinary TensorStudio.

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

- Keep tensors modest in `1.12.0`; kernels are CPU-first and still compact
  compared with mature high-performance ML runtimes.
- Prefer `float32` unless you need `float64`.
- Compare with NumPy in tests for expected numerical values.
- Watch shape errors carefully around broadcasting and matrix multiplication.
