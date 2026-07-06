# Quickstart

This guide takes you from installation to tensors, autograd, and a tiny neural
network.

## Requirements

- Python 3.10 or newer
- A C++20 compiler when building from source
- CMake and a supported Python packaging frontend

On Windows, install Microsoft C++ Build Tools or use a developer shell where
`cl` and `nmake` are available. On macOS, install Xcode Command Line Tools. On
Linux, install GCC or Clang plus CMake.

## Install From PyPI

```bash
python -m pip install tensorstudio
```

## Install From Source

```bash
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Validate the install:

```bash
python -c "import tensorstudio as ts; print(ts.__version__)"
pytest
```

## Create Tensors

```python
import tensorstudio as ts

x = ts.tensor([[1.0, 2.0], [3.0, 4.0]])
y = ts.ones((2, 2))
noise = ts.randn((2, 2), seed=123)

print(x.shape)       # (2, 2)
print(x.dtype)       # float32
print(noise.numpy()) # NumPy copy
```

## Run Operations

TensorStudio supports Python operators for arithmetic and matrix multiplication.

```python
z = (x + y) * 0.5
product = x @ y

print(z.tolist())
print(product.tolist())
print(x.reshape((4,)).tolist())
```

Broadcasting follows NumPy-style trailing dimension rules:

```python
a = ts.tensor([[1.0, 2.0], [3.0, 4.0]])
b = ts.tensor([10.0, 20.0])

print((a + b).tolist())
```

## Use Autograd

Set `requires_grad=True` on floating point tensors that should receive
gradients.

```python
x = ts.tensor([1.0, 2.0, 3.0], requires_grad=True)
loss = (x * x).mean()
loss.backward()

print(loss.item())
print(x.grad.tolist())
```

`backward()` can be called without an explicit gradient only for scalar outputs.

## Train A Tiny Model

```python
import tensorstudio as ts
from tensorstudio import nn, optim

x = ts.tensor([[0.0], [1.0], [2.0], [3.0]])
y = ts.tensor([[1.0], [3.0], [5.0], [7.0]])

model = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1))
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)

for step in range(100):
    optimizer.zero_grad()
    prediction = model(x)
    loss = loss_fn(prediction, y)
    loss.backward()
    optimizer.step()

print(loss.item())
```

## Next Steps

- Read [Tensor Basics](Usage/tensors.md) for shape and dtype details.
- Read [Autograd](Autograde/index.md) for graph and gradient behavior.
- Read [Development](development.md) before changing C++ or bindings.
