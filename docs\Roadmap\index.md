# Roadmap

TensorStudio `1.0.0` is a CPU-first foundation. This roadmap describes likely
directions without promising dates.

## Backend And Performance

- CUDA backend
- Improved CPU allocator
- SIMD kernels
- Multithreaded ops
- Better dtype dispatch
- Kernel benchmarking infrastructure

## Tensor API

- Axis reductions
- More view operations
- Advanced indexing
- Concatenation and stacking
- Dtype casting
- Device transfer API

## Autograd

- More operation gradients
- Non-scalar backward coverage
- Higher-order gradients
- Better graph lifecycle controls
- Safer in-place semantics

## Neural Networks

- Convolution ops
- Batch normalization
- Cross entropy loss
- Model zoo examples

## Interchange And Tooling

- ONNX import/export
- Richer serialization format
- Documentation site publishing
- More cibuildwheel targets
- Better error diagnostics

## Non-Goals For v1

- Distributed training
- Graph compiler
- Sparse tensors
- Production-scale training runtime
- Full parity with mature ML frameworks
