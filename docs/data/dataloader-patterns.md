# DataLoader Patterns

`DataLoader` turns a dataset into repeatable mini-batches. It is deliberately
simple in the current release: single-process, deterministic, and easy to read.

## Basic Training Loop

```python
import tensorstudio as ts
from tensorstudio import nn, optim
from tensorstudio.data import DataLoader, TensorDataset

x = ts.linspace(-1.0, 1.0, 64).reshape((64, 1))
y = 2.0 * x + 1.0
loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True, seed=7)

model = nn.Linear(1, 1)
optimizer = optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()

for features, target in loader:
    prediction = model(features)
    loss = loss_fn(prediction, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Deterministic Shuffling

Pass `seed=` when a test or example needs stable order. Reusing the same seed
gives the same first epoch order.

```python
loader = DataLoader(dataset, batch_size=16, shuffle=True, seed=123)
```

For production training, change the seed between runs or epochs if you need a
different order each time.

## `drop_last`

Use `drop_last=True` when every batch must have the same leading dimension:

```python
loader = DataLoader(dataset, batch_size=32, drop_last=True)
```

This is useful for model code that assumes fixed batch size, benchmark scripts,
or examples where a final small batch would distract from the main idea.

## Batch Design

Prefer batches shaped like:

- Features: `(batch, features)` for dense tabular data.
- Images: `(batch, channels, height, width)` for NCHW vision models.
- Labels: `(batch,)` for class indices or `(batch, outputs)` for regression.

Keep conversion to TensorStudio tensors at the dataset boundary. Training loops
should consume tensors directly rather than repeatedly converting lists or
NumPy arrays.

## Roadmap Fit

Future data work should add background workers, pinned-memory style staging for
future devices, streaming datasets, and richer dataset metadata. Those are not
implemented yet.
