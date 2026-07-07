# CPU Backend

TensorStudio `1.0.1` supports CPU tensors only.

## Device Abstraction

The C++ core includes a device abstraction so future backends can fit into the
same public model. Today, every tensor lives on CPU.

## What CPU Means In v1

- Storage is host memory.
- Operations run synchronously.
- Kernels are straightforward C++ loops.
- No explicit thread pool is used.
- No BLAS dependency is required.
- No SIMD-specific kernels are required.

## Portability

The code is intended to compile on:

- Linux with GCC or Clang
- macOS with Apple Clang
- Windows with MSVC

CMake and scikit-build-core coordinate the native extension build.

## GPU Future Work

CUDA and other accelerator backends are future work. Adding them would require:

- device-specific storage
- copy operations
- backend dispatch
- asynchronous execution rules
- kernel implementations
- expanded testing infrastructure

The current CPU implementation keeps those concerns out of the v1 release.
