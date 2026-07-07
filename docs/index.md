# TensorStudio

TensorStudio is a compact C++ tensor and autograd engine with a Python API for
learning, experimentation, and lightweight ML workloads.

`1.0.1` is a CPU-only stable API foundation for the tensor, autograd,
neural-network, optimizer, data, docs, packaging, and wheel workflows.

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
- Arithmetic, comparisons, matrix multiplication, all-element reductions, common
  activations, `sqrt`, `abs`, `clamp`, reshape, flatten, and 2D transpose.
- Reverse-mode autograd for the v1 operation set.
- Python `nn.Module`, parameters, linear layers, sequential models, activation
  modules, dropout, flatten, module introspection, and common losses.
- Python `optim.SGD`, `optim.Adam`, `optim.AdamW`, gradient clipping helpers,
  and small learning-rate schedulers.
- Minimal Windows-friendly `Dataset`, `TensorDataset`, and `DataLoader`.
- NumPy copy interop, pickle-based internal serialization, examples, tests,
  benchmarks, and GitHub Actions release workflows.

## Design Goals

TensorStudio prioritizes:

- Clear C++ loops and shape utilities over complex template metaprogramming.
- Honest errors for shape, dtype, device, and autograd issues.
- A Python API that feels familiar without pretending to be PyTorch-compatible.
- Source builds that work with MSVC, GCC, Clang, and Apple Clang.
- Wheels so end users can install without a compiler.

## Boundaries

TensorStudio does not currently include CUDA, Metal, distributed execution,
graph compilation, convolution layers, advanced indexing, sparse tensors, mixed
precision, or a high-performance kernel library. Benchmarks are rough local
references only; TensorStudio wins some small eager CPU cases but is not faster
than PyTorch, NumPy, TensorFlow, or JAX overall.

Serialization uses pickle. Loading pickle files from untrusted sources is
unsafe.

## Next Steps

- Start with [Quickstart](quickstart.md).
- Learn tensor semantics in [Tensors](tensors.md).
- Understand gradients in [Autograd](autograd.md).
- Build small models with [Neural Networks](nn.md).
- Use small datasets with [Data](data.md).
- Read [Publishing](publishing.md) before release work.
