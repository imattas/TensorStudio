# TensorStudio

[![CI](https://github.com/imattas/TensorStudio/actions/workflows/ci.yml/badge.svg)](https://github.com/imattas/TensorStudio/actions/workflows/ci.yml)
[![Wheels](https://github.com/imattas/TensorStudio/actions/workflows/wheels.yml/badge.svg)](https://github.com/imattas/TensorStudio/actions/workflows/wheels.yml)
[![PyPI](https://img.shields.io/pypi/v/tensorstudio.svg)](https://pypi.org/project/tensorstudio/)

TensorStudio is a compact C++ tensor and autograd engine with a Python API for
learning, experimentation, and lightweight ML workloads.

TensorStudio `1.0.0rc1` is a release candidate for a CPU-only stable API
foundation. It is eager-only, intentionally small, and not a replacement for
mature ML frameworks. The final `1.0.0` version should only be tagged after the
Windows, Linux, and macOS release checklist passes.

## Install

From PyPI, once release-candidate wheels are published:

```bash
python -m pip install tensorstudio
```

From a source checkout:

```bash
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Build source and wheel distributions:

```bash
python -m build
python -m twine check dist/*
```

End users should install wheels and should not need CMake. Source builds require
a C++20 compiler because the native extension is implemented in C++.

## Platform Setup

Windows is the primary release target. Use Python from python.org and install
Microsoft C++ Build Tools or Visual Studio with the Desktop development with C++
workload before building from source:

```powershell
python -m pip install -U pip
python -m pip install -e ".[dev]"
pytest -q
```

Linux source builds need GCC or Clang, CMake, and Python development headers:

```bash
python -m pip install -U pip
python -m pip install -e ".[dev]"
pytest -q
```

macOS source builds need Xcode Command Line Tools:

```bash
xcode-select --install
python -m pip install -U pip
python -m pip install -e ".[dev]"
pytest -q
```

## Quickstart

```python
import tensorstudio as ts

x = ts.tensor([[1.0, 2.0], [3.0, 4.0]])
y = ts.ones((2, 2))

print((x + y).tolist())
print((x @ y).numpy())
print(x.reshape((4,)).tolist())
```

## Tensor API

TensorStudio supports CPU tensors with `float32`, `float64`, `int32`, `int64`,
and `bool` dtypes.

```python
import tensorstudio as ts

ts.manual_seed(7)

a = ts.zeros((2, 3))
b = ts.rand((2, 3))
c = ts.eye(3)
d = ts.linspace(0.0, 1.0, 5)

print(a.shape, a.strides, a.device, a.is_contiguous)
print((b.clamp(0.2, 0.8) + 1).mean().item())
print(c.tolist(), d.tolist())
```

## Autograd

```python
import tensorstudio as ts

x = ts.tensor([1.0, 2.0, 3.0], requires_grad=True)
loss = (x * x).mean()
loss.backward()

print(x.grad.tolist())
```

Use `no_grad()` when you want eager computation without recording a graph:

```python
with ts.no_grad():
    y = x * 2
```

## Neural Networks

```python
import tensorstudio as ts
from tensorstudio import nn, optim

ts.manual_seed(0)

model = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1))
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
criterion = nn.MSELoss()

x = ts.tensor([[0.0], [1.0], [2.0], [3.0]])
y = ts.tensor([[1.0], [3.0], [5.0], [7.0]])

for _ in range(100):
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()

print(loss.item())
print(model.state_dict().keys())
```

## DataLoader

```python
import tensorstudio as ts
from tensorstudio.data import DataLoader, TensorDataset

dataset = TensorDataset(ts.arange(6).reshape((6, 1)), ts.arange(6))
loader = DataLoader(dataset, batch_size=2, shuffle=True, seed=42)

for features, targets in loader:
    print(features.shape, targets.shape)
```

The v1 release candidate DataLoader is intentionally single-process so it works
cleanly on Windows without multiprocessing setup.

## Performance

TensorStudio is optimized for small-to-medium CPU eager workloads, but
performance is still experimental. Benchmarks live in `benchmarks/` and can be
run locally:

```bash
python benchmarks/benchmark_report.py
```

On one Windows CPython 3.10 run for `1.0.0rc1`, TensorStudio beat NumPy on 14
small activation/reduction benchmark cases and lost on 75 NumPy-comparable
cases. The strongest local wins were small `sigmoid`, `sum`, and `mean` cases;
medium elementwise operations and matrix multiplication were much slower than
NumPy. See `benchmarks/results.md` for the full table, platform details, and
exact timings.

Do not treat these results as universal. TensorStudio does not currently claim
to be faster than NumPy, TensorFlow, PyTorch, or JAX overall.

## Save And Load

```python
import tensorstudio as ts
from tensorstudio import nn

model = nn.Linear(2, 1)
ts.save({"model": model.state_dict()}, "checkpoint.tsmodel")
checkpoint = ts.load("checkpoint.tsmodel")
```

Serialization uses pickle. Loading pickle files from untrusted sources is
unsafe because pickle can execute arbitrary code.

## Development

```bash
python -m pip install -e ".[dev,docs]"
ruff check .
mypy python/tensorstudio
pytest -q
python -m build
python -m twine check dist/*
```

The native extension module is `tensorstudio._C`, built with CMake, pybind11,
scikit-build-core, and C++20.

## Release Checklist

- `ruff check .` passes.
- `mypy python/tensorstudio` passes.
- `pytest -q` passes on Windows, Linux, and macOS.
- `python -m build` passes.
- `python -m twine check dist/*` passes.
- Benchmarks are generated and performance claims match the data.
- Clean wheel installs pass on Windows, Linux, and macOS.
- Clean sdist installs pass on Windows, Linux, and macOS.
- Examples run on all platforms.
- Docs match the implemented feature set.
- No PyPI tokens are committed or printed.
- TestPyPI is verified before a real PyPI release.

## Publishing

GitHub Actions build wheels with cibuildwheel. The publish workflow is designed
for PyPI trusted publishing with `id-token: write`; it should not hardcode PyPI
tokens or print secrets.

## Current Limitations

- CPU backend only.
- Eager execution only.
- No CUDA or Metal backend yet.
- No graph compiler or distributed runtime.
- No convolution layers yet.
- No sparse tensors or advanced indexing.
- Limited dtype casting.
- Experimental performance; benchmarks are local references only.
- Pickle serialization is for trusted TensorStudio objects only.

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
