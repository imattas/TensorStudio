# TensorStudio

[![CI](https://github.com/imattas/TensorStudio/actions/workflows/ci.yml/badge.svg)](https://github.com/imattas/TensorStudio/actions/workflows/ci.yml)
[![Wheels](https://github.com/imattas/TensorStudio/actions/workflows/wheels.yml/badge.svg)](https://github.com/imattas/TensorStudio/actions/workflows/wheels.yml)
[![PyPI](https://img.shields.io/pypi/v/tensorstudio.svg)](https://pypi.org/project/tensorstudio/)

TensorStudio is a compact C++ tensor and autograd engine with a Python API for
learning, experimentation, and lightweight ML workloads.

TensorStudio `1.14.0` is a CPU-only stable API foundation with native C++
threading, storage reuse, SIMD-friendly typed kernels, and optional
CBLAS/Accelerate matrix multiplication when available. It adds native stable
softmax/logsumexp, batched matrix multiplication, statistical reductions,
boolean reductions, seeded random distributions, and a hardened eager autograd
lifecycle. The neural-network layer now includes grouped/depthwise/1D/
transposed convolution, normalization layers, embeddings, richer activations,
initializers, additional losses, and model summaries. The project layer adds
dataset factories, deterministic train/validation splitting, metrics, callbacks,
multi-format configs, checkpoint resume helpers, and starter project templates.
The interchange layer adds richer NPZ metadata, optional SafeTensors weight
storage, ONNX metadata inspection, supported-subset ONNX import/execution, and
model metadata helpers. The hardware layer now exposes formal CPU/CUDA/Metal
device descriptors, backend availability reporting, device-aware storage
boundaries, explicit transfer APIs, and backend benchmark artifacts while
keeping the published wheels honest about CPU-only execution.
It is eager-only, intentionally compact, and not a replacement for mature ML
frameworks.

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
print(x[0, :].tolist())
print(x.unsqueeze(0).permute(1, 2, 0).shape)
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
labels = ts.randint((4,), low=0, high=3, seed=3)
mask = ts.bernoulli((2, 3), probability=0.25, seed=5)

print(a.shape, a.strides, a.device, a.is_contiguous)
print((b.clamp(0.2, 0.8) + 1).mean().item())
print(b.sum(axis=1).tolist())
print(ts.concat([b, b], axis=0).shape, b.astype("float64").dtype)
print(c.tolist(), d.tolist())
print(labels.tolist(), mask.any(axis=1).tolist())
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

Stable reductions and normalized probabilities are available as Tensor methods,
functional ops, and `tensorstudio.math` helpers:

```python
values = ts.tensor([[1000.0, 1001.0, 999.0], [1.0, 2.0, 3.0]])

print(ts.math.variance(values).item())
print(ts.math.std(values, axis=0).tolist())
print(ts.math.norm(values, ord=2).item())
print(values.softmax(axis=1).tolist())
print(ts.logsumexp(values, axis=1).tolist())
```

Batched matrix multiplication and a small documented `einsum` subset cover
common model and scientific-programming patterns:

```python
left = ts.randn((2, 3, 4), seed=1)
right = ts.randn((2, 4, 5), seed=2)

print((left @ right).shape)
print(ts.bmm(left, right).shape)
print(ts.einsum("bij,bjk->bik", left, right).shape)
```

## Autograd

```python
import tensorstudio as ts

x = ts.tensor([1.0, 2.0, 3.0], requires_grad=True)
loss = (x * x).mean()
loss.backward()

print(x.grad.tolist())
```

Reuse a graph explicitly when needed:

```python
loss = (x * x).sum()
loss.backward(retain_graph=True)
loss.backward()
```

Use `no_grad()` when you want eager computation without recording a graph:

```python
with ts.no_grad():
    y = x * 2
    x.zero_()
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

The neural-network surface also includes initialization helpers,
normalization layers, embeddings, grouped/depthwise/1D/transposed convolution,
adaptive/global pooling, richer activations, and model summaries:

```python
model = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1),
    nn.BatchNorm2d(8),
    nn.GELU(),
    nn.GlobalAvgPool2d(),
    nn.Flatten(),
    nn.Linear(8, 10),
)
nn.init.kaiming_uniform_(model[0].weight, nonlinearity="relu", seed=7)
print(nn.summary(model, input_shape=(1, 1, 28, 28))["total_parameters"])
```

## Vision

TensorStudio includes a practical computer-vision namespace for local image
classification workflows: Pillow-backed image IO, transform pipelines,
deterministic augmentations, `ImageFolder` datasets, metrics, image grids,
bounding-box drawing, and compact CNN classifiers running through native
Conv2d/pooling kernels.
The `1.14.0` vision surface includes batch-aware transforms, color jitter, random
resized crop, rotation, affine transforms, cutout, mixup, CutMix, detection
helpers, segmentation mask helpers, detection/segmentation folder datasets,
ResNet-style blocks, MobileNet-style depthwise blocks, a compact UNet, and
prediction/mask/feature-map visualization helpers.

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

## Data And Metrics

```python
import tensorstudio as ts
from tensorstudio.data import DataLoader, from_arrays, train_val_split

dataset = from_arrays([[0.0], [1.0], [2.0], [3.0]], [[1.0], [3.0], [5.0], [7.0]])
train_data, val_data = train_val_split(dataset, val_fraction=0.25, seed=42)
loader = DataLoader(train_data, batch_size=2, shuffle=True, seed=42)

for features, targets in loader:
    print(features.shape, targets.shape)

prediction = ts.tensor([[0.2, 0.8], [0.7, 0.3]])
target = ts.tensor([1, 0], dtype="int64")
print(ts.metrics.accuracy(prediction, target))
```

The v1 DataLoader is intentionally single-process so it works cleanly on
Windows without multiprocessing setup.

## Projects And Training

`tensorstudio.project` provides project folders, JSON/TOML/YAML config loading,
deterministic seeding, reusable trainers, validation loops, callbacks, safe NPZ
weight files, trusted full checkpoints, and generated starter templates:

```python
import tensorstudio as ts
from tensorstudio import nn, optim
from tensorstudio.data import DataLoader, from_arrays, train_val_split
from tensorstudio.project import (
    CSVLogger,
    CheckpointCallback,
    LrLogger,
    Project,
    ProjectConfig,
    Trainer,
    save_state_dict,
    seed_everything,
)

seed_everything(7)
dataset = from_arrays([[0.0], [1.0], [2.0], [3.0]], [[1.0], [3.0], [5.0], [7.0]])
train_data, val_data = train_val_split(dataset, val_fraction=0.25, seed=7)

model = nn.Linear(1, 1)
optimizer = optim.SGD(model.parameters(), lr=0.05)
trainer = Trainer(model, optimizer, nn.MSELoss(), metric_fn=ts.metrics.mean_squared_error)
project = Project("runs/linear", ProjectConfig(name="linear-regression", seed=7))

history = trainer.fit(
    DataLoader(train_data, batch_size=2),
    epochs=50,
    validation_loader=DataLoader(val_data, batch_size=1),
    callbacks=[
        LrLogger(),
        CSVLogger(project.logs_dir / "history.csv"),
        CheckpointCallback(project.checkpoints_dir / "best.tsmodel", save_best_only=True),
    ],
)
save_state_dict(model, project.checkpoint_path("weights"))
print(history.last)
```

## Hardware And Devices

TensorStudio `1.14.0` exposes explicit device descriptors and backend metadata.
The published wheels execute tensors on CPU only; CUDA and Metal descriptors are
available for feature checks and clear errors.

```python
import tensorstudio as ts

print(ts.available_devices())
print(ts.backend_info())

x = ts.ones((2, 2), device="cpu")
print(x.to("float64").dtype)
print(x.to_device("cpu").device)
print(ts.cuda_is_available())
```

Passing `device="cuda"` or `device="metal"` on CPU-only wheels raises
`DeviceError` instead of silently falling back.

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

Useful runtime diagnostics:

```python
import tensorstudio as ts

print(ts.performance_info())
ts.set_num_threads(4)
```

Run the loose local regression thresholds with:

```bash
python benchmark_all.py --check-thresholds
```

On one Windows CPython 3.10 development run reporting `1.14.0`, with
TensorStudio threads enabled, storage pooling enabled, SSE2 autovectorization
reported, and no BLAS provider found, TensorStudio beat NumPy on 6 local
benchmark cases and lost on 97 NumPy-comparable cases. JAX CPU dispatch was
available on that machine; TensorStudio won 38 local cases and lost 60. The
strongest local wins were the simple NumPy convolution/pooling reference loops
and some small JAX-dispatch-heavy eager cases. NumPy and JAX were faster for
many elementwise, reduction, matrix multiplication, larger activation, and
autograd workloads.
See `benchmarks/results.md` for the full table, platform details, and exact
timings.

Snapshot from that local run:

| operation | shape | TensorStudio | NumPy | JAX CPU dispatch | TS vs NumPy | TS vs JAX |
|---|---:|---:|---:|---:|---:|---:|
| `sigmoid` | `(32,)` | 0.0153 ms | 0.0073 ms | 0.0745 ms | 0.4756x | 4.8586x |
| `mean` | `(32,)` | 0.0156 ms | 0.0078 ms | 0.0114 ms | 0.4989x | 0.7318x |
| `sum_axis1` | `(16, 16)` | 0.0158 ms | 0.0026 ms | 0.0121 ms | 0.1664x | 0.7628x |
| `chain_relu` | `(128,)` | 0.0856 ms | 0.0055 ms | 0.0896 ms | 0.0642x | 1.0463x |
| `matmul` | `(256, 256)` | 2.0576 ms | 0.4624 ms | 0.2808 ms | 0.2247x | 0.1365x |
| `conv2d_3x3_padding1` | `(1, 1, 8, 8)` | 0.1832 ms | 1.2897 ms | 0.0998 ms | 7.0413x | 0.5449x |
| `max_pool2d_2x2` | `(1, 1, 16, 16)` | 0.0276 ms | 0.1571 ms | n/a | 5.6964x | n/a |
| `avg_pool2d_2x2` | `(1, 1, 16, 16)` | 0.0273 ms | 0.5435 ms | n/a | 19.9107x | n/a |
| `elementwise_backward` | `(1024,)` | 2.7089 ms | n/a | n/a | n/a | n/a |

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

For safe tensor-only weight files, install the optional SafeTensors extra:

```bash
python -m pip install "tensorstudio[safetensors]"
```

```python
ts.save_safetensors(model.state_dict(), "weights.safetensors")
state = ts.load_safetensors("weights.safetensors")
metadata = ts.inspect_model_metadata("weights.safetensors")
```

Supported ONNX graphs can be inspected and imported for TensorStudio execution:

```python
ts.export_onnx(model, "model.onnx", input_shape=(1, 2))
print(ts.inspect_onnx("model.onnx")["operators"])
imported = ts.import_onnx("model.onnx")
print(imported(ts.ones((1, 2))).shape)
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

The exporter supports `Linear`, `Conv2d`, grouped/depthwise `Conv2d`,
`ConvTranspose2d`, `Flatten`, `ReLU`, `Sigmoid`, `Tanh`, `MaxPool2d`, and
`AvgPool2d`. TensorStudio also includes ONNX metadata inspection and a
supported-subset importer for static graphs. It is not a general ONNX runtime.

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
- No CUDA or Metal tensor execution yet. CUDA/Metal device descriptors and
  build metadata exist, but unavailable accelerators raise `DeviceError`.
- Optional BLAS-backed matrix multiplication depends on the build environment
  exposing a compatible CBLAS/Accelerate interface; otherwise TensorStudio uses
  a portable C++ fallback.
- No graph compiler or distributed runtime.
- Convolution and pooling support are CPU-only. Native kernels include NCHW
  `conv2d`, grouped/depthwise convolution, `conv_transpose2d`, `max_pool2d`,
  `avg_pool2d`, and embedding lookup; they are not CUDA/cuDNN replacements.
- Vision covers local image-classification utilities, metrics, visualization,
  and compact CNNs. It is not an OpenCV replacement and does not include
  pretrained model zoos, detection/segmentation training stacks, video IO, or
  GPU image kernels yet.
- ONNX support covers export, metadata inspection, and import/execution for a
  limited static subset. It is not a full ONNX runtime.
- Reductions support all-element, single-axis, and tuple/list-axis reductions
  for `sum`, `mean`, `max`, and `min`.
- Arg reductions support all-element flat indices or one axis at a time for
  `argmax` and `argmin`.
- Selection helpers `where`, `maximum`, and `minimum` are native C++ tensor ops
  with broadcasting and autograd support for floating-point branches.
- Basic integer/slice indexing is supported as native C++ views with autograd
  scatter-back. Advanced list, tensor, and boolean-mask indexing are not
  implemented yet.
- Dtype casting is basic and does not include a full promotion/casting policy.
- Experimental performance; benchmarks are local references only.
- Pickle serialization is for trusted TensorStudio objects only.

## Roadmap

- CUDA and Metal execution backends
- Graph/JIT mode
- Broader convolution ops, adaptive/global pooling, and image-model examples
- Richer dataset utilities
- Model zoo examples
- Broader ONNX operator coverage
- Runtime-dispatched SIMD kernels
- Better non-BLAS matrix multiplication tiling
- More threaded backward kernels

## License

TensorStudio is licensed under the MIT License.
