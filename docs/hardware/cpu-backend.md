# CPU Backend

TensorStudio `1.9.0` supports CPU tensors only, with a stronger native runtime
than earlier releases.

## Device Abstraction

The C++ core includes a device abstraction so future backends can fit into the
same public model. Today, every tensor lives on CPU.

## What CPU Means In v1

- Storage is host memory.
- Operations run synchronously.
- Hot kernels are implemented in C++.
- Large contiguous elementwise operations, reductions, matrix multiplication,
  convolution, and pooling forward paths can use a small native thread pool.
- Contiguous `float32` and `float64` elementwise kernels use typed loops that
  avoid unnecessary `double` conversion and are friendly to compiler
  autovectorization.
- Contiguous `float32` and `float64` 2D matrix multiplication uses CBLAS or
  Accelerate when the source build has compatible BLAS support; otherwise it
  uses the portable C++ fallback.
- A bounded storage reuse pool reduces repeated allocation cost while keeping
  previous zero-initialization behavior.

## Runtime Controls

```python
import tensorstudio as ts

print(ts.performance_info())
ts.set_num_threads(4)
print(ts.get_num_threads())
```

Environment variables:

- `TENSORSTUDIO_NUM_THREADS`: preferred native worker count.
- `TENSORSTUDIO_DISABLE_THREADS=1`: force single-thread execution.
- `TENSORSTUDIO_DISABLE_STORAGE_POOL=1`: disable storage reuse.
- `TENSORSTUDIO_STORAGE_POOL_MAX_BLOCK_BYTES`: maximum reusable block size.

The defaults favor portability. Very small tensors often run on one thread
because thread scheduling overhead can dominate the math.

## BLAS

TensorStudio has an optional BLAS path, not a hard BLAS dependency. Source
builds try to find a BLAS library with CMake. The optimized path is enabled
only when a usable CBLAS-compatible interface is available:

- macOS can use Accelerate.
- Linux can use OpenBLAS, BLIS, MKL, or another CBLAS-compatible package when
  headers and libraries are visible to CMake.
- Windows can use OpenBLAS or MKL through a toolchain such as vcpkg or a
  system install that exposes `cblas.h`.

Official wheels may use the portable fallback if the wheel build environment
does not provide a redistributable CBLAS setup. Check `ts.performance_info()`
instead of assuming BLAS is present.

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
