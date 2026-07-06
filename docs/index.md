# TensorStudio

TensorStudio is a compact C++ tensor and autograd engine with a Python API for
learning, experimentation, and lightweight ML workloads.

The project is intentionally small, readable, and direct. The C++ core owns
tensor storage, dtype handling, shape validation, broadcasting, eager operations,
and reverse-mode automatic differentiation. The Python package provides a
friendly API, neural network modules, optimizers, serialization helpers, tests,
examples, benchmarks, and release tooling.

## Project Status

TensorStudio v0.1.1 is experimental. It is useful for learning how tensor
frameworks are assembled, for small experiments, and for lightweight workloads.
It is not a replacement for mature production ML frameworks.

## What Is Included

- CPU tensor storage with shared storage, shape, stride, offset, dtype, and
  autograd metadata.
- Dtypes: `float32`, `float64`, `int32`, `int64`, and `bool`.
- NumPy-style broadcasting for binary elementwise operations.
- Matrix multiplication for 2D tensors.
- Scalar reductions, common activations, reshape, flatten, and 2D transpose.
- Reverse-mode autograd for the v0.1.x operation set.
- Python `nn.Module`, `Parameter`, `Linear`, `Sequential`, activations, and
  `MSELoss`.
- Python `optim.SGD` and `optim.Adam`.
- NumPy copy interop, pickle-based internal serialization, examples, tests, and
  benchmark scripts.

## Design Goals

TensorStudio prioritizes:

- Clear C++ implementation over heavy template machinery.
- Eager execution and readable control flow.
- Small public API surface that feels familiar to PyTorch/TensorFlow users.
- Honest documentation of limitations and edge cases.
- Packaging that is realistic for PyPI, CI, and wheel builds.

## Current Boundaries

TensorStudio v0.1.1 is CPU-only. It has no CUDA backend, no graph compiler, no
distributed runtime, no advanced indexing, no sparse tensors, and limited dtype
casting. Serialization uses pickle, so loading untrusted files is unsafe.

## Where To Go Next

- Start with [Quickstart](quickstart.md).
- Learn tensor behavior in [Tensor Basics](Usage/tensors.md).
- Understand gradients in [Autograd](Autograde/index.md).
- Build small models with [Neural Network Modules](Neural%20Networks/modules.md).
- Prepare releases with [Publishing](publishing.md).
