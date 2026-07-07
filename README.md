# TensorStudio

[![CI](https://github.com/imattas/TensorStudio/actions/workflows/ci.yml/badge.svg)](https://github.com/imattas/TensorStudio/actions/workflows/ci.yml)
[![Wheels](https://github.com/imattas/TensorStudio/actions/workflows/wheels.yml/badge.svg)](https://github.com/imattas/TensorStudio/actions/workflows/wheels.yml)
[![PyPI](https://img.shields.io/pypi/v/tensorstudio.svg)](https://pypi.org/project/tensorstudio/)

TensorStudio is a compact C++ tensor and autograd engine with a Python API for
learning, experimentation, and lightweight ML workloads.

TensorStudio `1.3.6` is a CPU-only stable API foundation. It is eager-only,
intentionally small, and not a replacement for mature ML frameworks.

## Install

From PyPI:

```bash
python -m pip install tensorstudio
```

From a source checkout:

```bash
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Install optional extras for ONNX export and Pillow-backed image inputs:

```bash
python -m pip install "tensorstudio[onnx,vision]"
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
print(b.sum(axis=1).tolist())
print(ts.concat([b, b], axis=0).shape, b.astype("float64").dtype)
print(c.tolist(), d.tolist())
print(ts.zeros_like(b).shape, ts.randn_like(b, seed=11).dtype)
```

Arithmetic promotion is explicit and inspectable:

```python
print(ts.promote_types("int32", "float32"))        # float32
print(ts.result_type("int64", "int32", op="div")) # float32
print(ts.result_type("int64", "float32", op="gt")) # bool
```

## Advanced Math

Native C++ elementwise math includes trigonometric functions and numerically
useful helpers with autograd support:

```python
import tensorstudio as ts

x = ts.tensor([0.1, 0.2, 0.3], requires_grad=True)
y = ts.sin(x) + x.cos() + x.log1p() + x.rsqrt()
loss = y.mean()
loss.backward()

print(loss.item())
print(x.grad.tolist())
```

Higher-level helpers live in `tensorstudio.math`:

```python
values = ts.tensor([[1.0, 2.0], [3.0, 4.0]])

print(ts.math.variance(values).item())
print(ts.math.std(values, axis=0).tolist())
print(ts.math.norm(values, ord=2).item())
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
scheduler = optim.StepLR(optimizer, step_size=50, gamma=0.5)
criterion = nn.MSELoss()

x = ts.tensor([[0.0], [1.0], [2.0], [3.0]])
y = ts.tensor([[1.0], [3.0], [5.0], [7.0]])

for _ in range(100):
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optim.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()
    scheduler.step()

print(loss.item())
print(model.state_dict().keys())
print(model.parameter_count())
```

## Vision

TensorStudio includes a practical computer-vision namespace for local image
classification workflows: Pillow-backed image IO, transform pipelines,
deterministic augmentations, `ImageFolder` datasets, metrics, image grids,
bounding-box drawing, and compact CNN classifiers running through native
Conv2d/pooling kernels.

```python
import numpy as np
import tensorstudio as ts
from tensorstudio import nn, optim

transform = ts.vision.Compose(
    [
        ts.vision.Resize((8, 8)),
        ts.vision.ToTensor(),
        ts.vision.Normalize(0.5, 0.5),
    ]
)
image = np.zeros((8, 8, 3), dtype=np.uint8)
x = transform(image).reshape((1, 3, 8, 8))

model = ts.vision.ImageClassifier((3, 8, 8), num_classes=2, channels=(4,))
target = ts.tensor([1], dtype="int64")
optimizer = optim.SGD(model.parameters(), lr=0.01)

optimizer.zero_grad()
loss = nn.CrossEntropyLoss()(model(x), target)
loss.backward()
optimizer.step()
print(ts.vision.accuracy(model(x), target))
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

The v1 DataLoader is intentionally single-process so it works cleanly on
Windows without multiprocessing setup.

## Projects And Training

`tensorstudio.project` provides project folders, JSON config, reusable trainers,
safe NPZ weight files, and trusted full checkpoints:

```python
import tensorstudio as ts
from tensorstudio import nn, optim
from tensorstudio.data import DataLoader, TensorDataset
from tensorstudio.project import Project, ProjectConfig, Trainer, save_state_dict

x = ts.tensor([[0.0], [1.0], [2.0], [3.0]])
y = ts.tensor([[1.0], [3.0], [5.0], [7.0]])

model = nn.Linear(1, 1)
loader = DataLoader(TensorDataset(x, y), batch_size=2)
trainer = Trainer(model, optim.SGD(model.parameters(), lr=0.05), nn.MSELoss())
project = Project("runs/linear", ProjectConfig(name="linear-regression", seed=7))

history = trainer.fit(loader, epochs=50)
save_state_dict(model, project.checkpoint_path("weights"))
print(history.last)
```

## Performance

TensorStudio is optimized for small-to-medium CPU eager workloads, but
performance is still experimental. Benchmarks live in `benchmarks/` and can be
run locally:

```bash
python benchmark_all.py
python benchmarks/benchmark_report.py
```

`benchmark_all.py` writes `benchmarks/results.md` and includes explicit win
columns for NumPy, TensorFlow, PyTorch, and JAX when those libraries are
available locally.

On one Windows CPython 3.10 development run reporting `1.2.0`, TensorStudio
beat NumPy on 26 small operation benchmark cases and lost on 77
NumPy-comparable cases. Against PyTorch CPU `2.12.1+cpu`, TensorStudio won 75
local cases and lost 33. The strongest local wins were small eager operations,
small contiguous axis reductions, and the simple NumPy convolution/pooling
references where framework dispatch or Python loops dominate; larger matrix
multiplication, PyTorch convolution and pooling, larger axis reductions, larger
transcendental activations, and larger autograd workloads remain faster in
PyTorch and NumPy.
See `benchmarks/results.md` for the full table, platform details, and exact
timings.

Snapshot from that local run:

| operation | shape | TensorStudio | NumPy | PyTorch CPU | TS vs NumPy | TS vs PyTorch |
|---|---:|---:|---:|---:|---:|---:|
| `sigmoid` | `(32,)` | 0.0017 ms | 0.0038 ms | 0.0614 ms | 2.1691x | 35.3154x |
| `mean` | `(32,)` | 0.0029 ms | 0.0131 ms | 0.0186 ms | 4.5638x | 6.4709x |
| `sum_axis1` | `(16, 16)` | 0.0027 ms | 0.0029 ms | 0.0085 ms | 1.0894x | 3.1408x |
| `chain_relu` | `(128,)` | 0.0112 ms | 0.0051 ms | 0.0726 ms | 0.4575x | 6.4533x |
| `matmul` | `(256, 256)` | 4.4121 ms | 0.3779 ms | 0.1669 ms | 0.0856x | 0.0378x |
| `conv2d_3x3_padding1` | `(1, 1, 8, 8)` | 0.2374 ms | 1.7112 ms | 0.0186 ms | 7.2091x | 0.0784x |
| `max_pool2d_2x2` | `(1, 1, 16, 16)` | 0.0237 ms | 0.2764 ms | 0.0082 ms | 11.6685x | 0.3462x |
| `avg_pool2d_2x2` | `(1, 1, 16, 16)` | 0.0198 ms | 0.7431 ms | 0.0073 ms | 37.4930x | 0.3662x |
| `elementwise_backward` | `(1024,)` | 3.0196 ms | n/a | 0.2254 ms | n/a | 0.0746x |

Speedup is `competitor median / TensorStudio median`, so values above `1.0x`
favor TensorStudio.

Do not treat these results as universal. TensorStudio does not claim to be
faster than NumPy, TensorFlow, PyTorch, or JAX overall.

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

For safer tensor and `state_dict` interchange, use TensorStudio's non-pickle
NPZ helpers:

```python
state = model.state_dict()
ts.save_npz(state, "weights.tsnpz")
model.load_state_dict(ts.load_npz("weights.tsnpz"))
```

## ONNX Export

TensorStudio can export a supported `nn.Sequential` graph to ONNX when the
optional `onnx` extra is installed:

```python
import tensorstudio as ts
from tensorstudio import nn

model = nn.Sequential(
    nn.Conv2d(1, 2, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(2 * 2 * 2, 3),
)

ts.export_onnx(model, "classifier.onnx", input_shape=(1, 1, 4, 4))
```

The exporter supports `Linear`, `Conv2d`, `Flatten`, `ReLU`, `Sigmoid`,
`Tanh`, `MaxPool2d`, and `AvgPool2d`. It is an exporter, not an ONNX runtime or
importer.

## Development

```bash
python -m pip install -e ".[dev,docs]"
python test_all.py --skip-build
ruff check .
mypy python/tensorstudio
pytest -q
python -m build
python -m twine check dist/*
```

The native extension module is `tensorstudio._C`, built with CMake, pybind11,
scikit-build-core, and C++20.

## Release Checklist

- `python test_all.py` passes locally.
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
- No BLAS-backed matrix multiplication yet.
- No graph compiler or distributed runtime.
- Convolution and pooling support are currently limited to CPU NCHW
  `conv2d`, `max_pool2d`, and `avg_pool2d` style workloads.
- Vision covers local image-classification utilities, metrics, visualization,
  and compact CNNs. It is not an OpenCV replacement and does not include
  pretrained model zoos, detection/segmentation training stacks, video IO, or
  GPU image kernels yet.
- ONNX support is export-only for a limited set of TensorStudio modules.
- Reductions support all-element, single-axis, and tuple/list-axis reductions
  for `sum`, `mean`, `max`, and `min`.
- Arg reductions support all-element flat indices or one axis at a time for
  `argmax` and `argmin`.
- Selection helpers `where`, `maximum`, and `minimum` are native C++ tensor ops
  with broadcasting and autograd support for floating-point branches.
- No sparse tensors or advanced indexing.
- Dtype casting is basic and does not include a full promotion/casting policy.
- Experimental performance; benchmarks are local references only.
- Pickle serialization is for trusted TensorStudio objects only.

## Roadmap

- CUDA backend
- Graph/JIT mode
- Broader convolution ops, adaptive/global pooling, and image-model examples
- Richer dataset utilities
- Model zoo examples
- ONNX import and broader export coverage
- Improved memory allocator
- SIMD kernels
- Multithreaded ops

## License

TensorStudio is licensed under the MIT License.
