# TensorStudio

TensorStudio is a compact C++ tensor and autograd engine with a Python API for
learning, experimentation, and lightweight ML workloads.

`1.3.2` is a CPU-only stable API foundation for the tensor, autograd,
neural-network, optimizer, data, project, serialization, ONNX export, vision,
docs, packaging, and wheel workflows.

## Status

TensorStudio is:

- CPU-only.
- Eager-only.
- Built around a native `tensorstudio._C` extension.
- Suitable for learning, experimentation, and lightweight ML workloads.
- Not intended to replace or broadly outperform mature production ML frameworks.

## What Is Included

- C++20 tensor storage with shared storage, shape, strides, offset, dtype,
  device, and autograd metadata.
- Dtypes: `float32`, `float64`, `int32`, `int64`, and `bool`.
- Tensor creation helpers: `tensor`, `from_numpy`, `zeros`, `ones`, `full`,
  `empty`, `rand`, `randn`, `arange`, `eye`, `linspace`, and matching
  `*_like` helpers.
- NumPy-style broadcasting for binary elementwise operations.
- Arithmetic, comparisons, matrix multiplication, all-element and single-axis
  reductions, common activations, trigonometric functions, inverse
  trigonometric functions, `log1p`, `sqrt`, `rsqrt`, `abs`, `clamp`, CPU NCHW
  `conv2d`, `max_pool2d`, `avg_pool2d`, reshape, flatten, and 2D transpose.
- Higher-level `tensorstudio.math` helpers for variance, standard deviation,
  norms, square, and reciprocal.
- Basic dtype casting plus native `concat` and `stack`.
- Reverse-mode autograd for the v1 operation set.
- Python `nn.Module`, parameters, linear, convolution, and pooling layers,
  sequential models, activation modules, dropout, flatten, module introspection,
  and common losses including multiclass cross entropy.
- Python `optim.SGD`, `optim.Adam`, `optim.AdamW`, gradient clipping helpers,
  and small learning-rate schedulers.
- Minimal Windows-friendly `Dataset`, `TensorDataset`, and `DataLoader`.
- Project utilities for JSON configs, run folders, reusable trainers, safe NPZ
  state-dict checkpoints, and trusted full checkpoints.
- NumPy copy interop, pickle-based internal serialization, non-pickle NPZ
  tensor/state_dict files, limited ONNX export, vision IO, transforms, datasets,
  metrics, visualization, compact CNN helpers, examples, tests, benchmarks, and
  GitHub Actions release workflows.

## Design Goals

TensorStudio prioritizes:

- Clear C++ loops and shape utilities over complex template metaprogramming.
- Honest errors for shape, dtype, device, and autograd issues.
- A Python API that feels familiar without pretending to be PyTorch-compatible.
- Source builds that work with MSVC, GCC, Clang, and Apple Clang.
- Wheels so end users can install without a compiler.

## Boundaries

TensorStudio does not currently include CUDA, Metal, distributed execution,
graph compilation, grouped convolution, adaptive/global pooling, advanced
indexing, sparse tensors, mixed precision, tuple-axis reductions, a high
performance kernel library, pretrained vision model zoo, detection/segmentation
training stack, video IO, an ONNX runtime, or ONNX import. Benchmarks are rough
local references only; TensorStudio wins some small eager CPU cases but is not
faster than PyTorch, NumPy, TensorFlow, or JAX overall.

Pickle serialization is available for trusted internal objects. Prefer
`save_npz`/`load_npz` for portable tensor and state_dict files.

## Next Steps

- Start with [Quickstart](quickstart.md).
- Learn tensor semantics in [Tensors](Usage/tensors.md).
- Understand gradients in [Autograd](Autograde/autograd.md).
- Build small models with [Neural Networks](Neural%20Networks/nn.md).
- Organize complete runs with [Projects](Projects/index.md).
- Build image classifiers with [Vision](Vision/index.md).
- Export supported module stacks with [ONNX Interchange](Usage/interchange.md).
- Use small datasets with [Data](Usage/data.md).
- Read [Publishing](publishing.md) before release work.
