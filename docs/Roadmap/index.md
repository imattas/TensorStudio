# Roadmap

TensorStudio `1.1.0` is a CPU-first foundation. This roadmap describes likely
directions without promising dates.

## Backend And Performance

- CUDA backend
- Improved CPU allocator
- SIMD kernels
- Multithreaded ops
- Better dtype dispatch
- Kernel benchmarking infrastructure

## Tensor API

- Tuple-axis reductions and arg reductions
- More view operations
- Advanced indexing
- More tensor layout transforms
- Richer dtype promotion and casting policy
- Device transfer API

## Autograd

- More operation gradients
- Non-scalar backward coverage
- Higher-order gradients
- Better graph lifecycle controls
- Safer in-place semantics

## Neural Networks

- Broader convolution ops beyond the initial CPU NCHW `conv2d`/`nn.Conv2d`
- Adaptive pooling, global pooling, and more image-model building blocks
- Batch normalization
- Label smoothing and additional classification losses
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
