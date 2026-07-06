# TensorStudio

[![CI](https://github.com/tensorstudio/tensorstudio/actions/workflows/ci.yml/badge.svg)](https://github.com/tensorstudio/tensorstudio/actions/workflows/ci.yml)
[![Wheels](https://github.com/tensorstudio/tensorstudio/actions/workflows/wheels.yml/badge.svg)](https://github.com/tensorstudio/tensorstudio/actions/workflows/wheels.yml)
[![PyPI](https://img.shields.io/pypi/v/tensorstudio.svg)](https://pypi.org/project/tensorstudio/)

TensorStudio is a compact C++ tensor and autograd engine with a Python API for
learning, experimentation, and lightweight ML workloads.

TensorStudio is experimental v0.1.1 software. It is CPU-only, eager-only, and
intentionally small enough to read and modify.

## Install

From a source checkout:

```bash
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Build source and wheel distributions:

```bash
python -m build
```

## Quickstart

```python
import tensorstudio as ts

x = ts.tensor([[1.0, 2.0], [3.0, 4.0]])
y = ts.ones((2, 2))

print((x + y).tolist())
print((x @ y).numpy())
```

## Autograd

```python
import tensorstudio as ts

x = ts.tensor([1.0, 2.0, 3.0], requires_grad=True)
loss = ((x * x).sum())
loss.backward()

print(x.grad.tolist())  # [2.0, 4.0, 6.0]
```

## Neural Networks

```python
import tensorstudio as ts
from tensorstudio import nn, optim

model = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1))
optimizer = optim.SGD(model.parameters(), lr=0.05)
criterion = nn.MSELoss()

x = ts.tensor([[0.0], [1.0], [2.0], [3.0]])
y = ts.tensor([[1.0], [3.0], [5.0], [7.0]])

for _ in range(50):
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()

print(loss.item())
```

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
mypy python/tensorstudio
python -m build
```

The native extension is built with CMake, pybind11, scikit-build-core, and C++20.

## Documentation

The documentation lives in `docs/` and is configured with MkDocs:

```bash
python -m pip install -e ".[docs]"
mkdocs serve
mkdocs build
```

The docs cover tensor semantics, broadcasting, NumPy interop, autograd, neural
network modules, training loops, CPU backend details, benchmarks, development,
publishing, and the project roadmap.

## Publishing

Releases are intended to be built by GitHub Actions. The publish workflow uses
PyPI trusted publishing through `pypa/gh-action-pypi-publish`; do not commit PyPI
API tokens.

## Current Limitations

- CPU backend only
- Eager execution only
- No CUDA kernels or mixed precision
- No graph compiler or distributed training
- Limited dtype casting and no advanced indexing
- No sparse tensors
- Pickle serialization is for trusted TensorStudio objects only

## Roadmap

- CUDA backend
- Graph/JIT mode
- Convolution ops
- Dataset utilities
- Model zoo examples
- ONNX import/export
- Improved memory allocator
- SIMD kernels
- Multithreaded ops

## License

TensorStudio is licensed under the MIT License.
