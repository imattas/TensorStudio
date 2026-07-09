# CPU Backend

TensorStudio `2.1.0` supports CPU tensors only.

## Device Abstraction

The C++ core includes a device abstraction so future backends can fit into the
same public model. Today, every executable tensor lives on CPU. CUDA, Metal,
and registered plugin descriptors can be inspected through the hardware API,
but non-CPU storage allocation is rejected until real backend storage and
kernels exist.

See [Backend Architecture](backend-architecture.md) for the registry and
placement-diagnostics design.

## What CPU Means In v1

- Storage is host memory.
- Operations run synchronously.
- Kernels are native C++ loops; large contiguous elementwise and full-reduction
  kernels use a conservative thread dispatcher.
- Allocator metadata is exposed through `backend_allocator_info("cpu")`.
- Native storage allocation counters are exposed through
  `storage_telemetry()`.
- Physical-device metadata is exposed through `backend_device_properties("cpu")`.
- The CPU exposes one logical device through `logical_device_info("cpu")`.
- Operation contracts are exposed through `backend_op_info()`.
- CPU kernel capability metadata is exposed through `backend_kernel_info("cpu")`.
- Eager placement can be inspected through `backend_execution_plan(op, "cpu")`.
- CPU-to-CPU tensor movement is exposed through `Tensor.to_device("cpu")` and
  `ts.to_device(tensor, "cpu", copy=True)`.
- No persistent thread pool is used.
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
