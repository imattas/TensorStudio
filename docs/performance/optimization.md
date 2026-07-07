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
Creating many temporary tensors can be more expensive than the math.
TensorStudio `1.6.0` includes a bounded native storage pool for repeated
allocation sizes. It reuses blocks after tensors release their final reference
and zeroes reused storage before handing it back to preserve existing behavior.
Prefer fused kernels only when the behavior is stable and tests are strong.

Strides:
Contiguous tensors are easiest to optimize. Strided views need careful indexing
and should be benchmarked separately.

Broadcasting:
Broadcast setup should happen once per operation. Avoid recomputing shape
metadata inside the innermost loop.

Threading:
Use the native thread pool only when the tensor is large enough to repay
scheduling overhead. Tiny eager operations should usually remain single-thread.

SIMD:
The first SIMD-oriented path is compiler-friendly typed `float32` and
`float64` contiguous loops. Do not require CPU-specific instruction sets in
portable wheels unless runtime dispatch is in place.

BLAS:
Use CBLAS/Accelerate for compatible contiguous matrix multiplication when the
build environment exposes it. Keep a tested portable fallback because many
source and wheel environments do not have a redistributable BLAS stack.

Autograd:
Backward formulas should reuse existing native primitives when that keeps the
implementation correct and maintainable. For hot paths, custom backward kernels
may be worth adding later.

## Near-Term Performance Priorities

- Wider typed kernels for more math operations.
- Better matrix multiplication tiling when BLAS is unavailable.
- Fewer temporary tensors in common backward formulas.
- Runtime-dispatched SIMD instruction paths after portability hardening.
- More threaded backward kernels once race-free accumulation paths are pinned
  down by tests.
