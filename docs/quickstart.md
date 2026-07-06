# Quickstart

This guide covers installation, tensor creation, operations, autograd, a tiny
neural network, data loading, and serialization.

## Requirements

- Python 3.10 or newer.
- A C++20 compiler only when building from source.
- NumPy, installed automatically as a runtime dependency.

Prebuilt wheels are the preferred installation path. Source builds use CMake,
pybind11, and scikit-build-core.

## Install

From PyPI, after release-candidate wheels are published:

```bash
python -m pip install tensorstudio
```

From a source checkout:

```bash
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Validate the install:

```bash
python -c "import tensorstudio as ts; import tensorstudio._C; print(ts.__version__)"
pytest -q
```

## Create Tensors

```python
import tensorstudio as ts

x = ts.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y = ts.ones((2, 2))
z = ts.linspace(0.0, 1.0, 5)

print(x.shape)          # (2, 2)
print(x.strides)        # (2, 1)
print(x.dtype)          # float32
print(x.device)         # cpu
print(x.is_contiguous)  # True
print(z.tolist())
```

## Creation Helpers

```python
ts.zeros((2, 3))
ts.empty((2, 3))
ts.ones((2, 3), dtype="float64")
ts.full((2, 3), 7.0)
ts.rand((2, 3), seed=123)
ts.randn((2, 3), seed=123)
ts.arange(0, 10, 2)
ts.eye(3)
ts.linspace(-1.0, 1.0, 5)
```

`manual_seed(seed)` seeds TensorStudio's process-local C++ random generator:

```python
ts.manual_seed(42)
sample = ts.randn((2, 2))
```

## Operations

TensorStudio supports Python operators for arithmetic and matrix multiplication:

```python
a = ts.tensor([[1.0, 2.0], [3.0, 4.0]])
b = ts.tensor([10.0, 20.0])

print((a + b).tolist())
print((a * 2 - 1).tolist())
print((a @ ts.eye(2)).tolist())
print(a.T.tolist())
```

Supported math includes `sum`, `mean`, `max`, `min`, `relu`, `sigmoid`, `tanh`,
`exp`, `log`, `sqrt`, `abs`, and `clamp`.

```python
x = ts.tensor([-2.0, -1.0, 0.0, 4.0])
print(x.abs().tolist())
print(x.clamp(-1.0, 2.0).tolist())
```

## Autograd

```python
x = ts.tensor([1.0, 2.0, 3.0], requires_grad=True)
loss = (x * x).mean()
loss.backward()

print(loss.item())
print(x.grad.tolist())
```

For non-scalar outputs, pass an explicit gradient:

```python
y = x * 2.0
y.backward(ts.ones(y.shape))
```

Disable graph recording with `no_grad()`:

```python
with ts.no_grad():
    doubled = x * 2.0
```

## Neural Network

```python
from tensorstudio import nn, optim

ts.manual_seed(0)

x = ts.tensor([[0.0], [1.0], [2.0], [3.0]])
y = ts.tensor([[1.0], [3.0], [5.0], [7.0]])

model = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1))
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

for _ in range(100):
    optimizer.zero_grad()
    prediction = model(x)
    loss = loss_fn(prediction, y)
    loss.backward()
    optimizer.step()

print(loss.item())
```

## DataLoader

```python
from tensorstudio.data import DataLoader, TensorDataset

dataset = TensorDataset(ts.arange(6).reshape((6, 1)), ts.arange(6))
loader = DataLoader(dataset, batch_size=2, shuffle=True, seed=7)

for features, targets in loader:
    print(features.tolist(), targets.tolist())
```

The DataLoader is single-process by design for the v1 release candidate.

## Save And Load

```python
checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
ts.save(checkpoint, "checkpoint.tsmodel")
loaded = ts.load("checkpoint.tsmodel")
```

Only load TensorStudio pickle files from trusted sources.
