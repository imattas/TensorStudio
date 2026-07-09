# Optimization Guide

TensorStudio should prefer native C++ for work that loops over tensor elements
or runs inside training steps. Python should provide names, composition, and
ergonomics.

## Native First Targets

Good candidates for C++ implementation:

- Elementwise math.
- Broadcasting.
- Reductions.
- Matrix kernels.
- Convolution and pooling kernels.
- Autograd formulas for primitive ops.
- Tensor serialization primitives.

Keep Python implementations for orchestration, optional integrations, and code
that does not dominate runtime.

## Kernel Checklist

Before adding an optimized kernel:

1. Define shape and dtype behavior.
2. Add NumPy parity tests where possible.
3. Add autograd tests if the op is differentiable.
4. Benchmark small, medium, and large inputs.
5. Document known wins and losses.

## Common Bottlenecks

Allocation:
Creating many temporary tensors can be more expensive than the math. Prefer
fused kernels only when the behavior is stable and tests are strong.

Strides:
Contiguous tensors are easiest to optimize. Strided views need careful indexing
and should be benchmarked separately.

Matrix multiplication:
Contiguous `float32` 2D matmul keeps the faster accumulator strategy for small
and medium matrices and uses a native cache-blocked path for larger workloads.
Mixed dtypes, strided views, batched matmul, and optional BLAS-backed builds
remain future optimization work.

Broadcasting:
Broadcast setup should happen once per operation. Avoid recomputing shape
metadata inside the innermost loop.

Threading:
Large contiguous unary, binary, and full-reduction CPU kernels use a
conservative C++ thread dispatcher. Small tensors stay single-threaded so
thread startup does not dominate interactive workloads.

Autograd:
Backward formulas should reuse existing native primitives when that keeps the
implementation correct and maintainable. For hot paths, custom backward kernels
may be worth adding later.

## Near-Term Performance Priorities

- Faster reductions for larger contiguous and strided tensors.
- Better matrix multiplication for larger square and rectangular cases.
- Fewer temporary tensors in common backward formulas.
- Optional SIMD loops for simple contiguous elementwise kernels.
- Broader threaded CPU kernels, especially backward, convolution, pooling, and
  normalization paths.
