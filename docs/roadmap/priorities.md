# Priorities

The roadmap is ordered around foundations first. TensorStudio should become
broader only when the core runtime stays understandable, tested, and fast
enough to justify the added surface area.

## Priority 1: Correctness

- Shape validation with clear errors.
- DType promotion that matches documented rules.
- Autograd formulas tested against finite differences where practical.
- Serialization that round-trips without silent data loss.

## Priority 2: Native Performance

- Move hot tensor loops into C++.
- Improve contiguous reductions.
- Improve matrix multiplication.
- Add SIMD and threading only after scalar kernels are stable.
- Keep benchmark losses visible.

## Priority 3: Model Coverage

- More neural-network modules.
- Better vision layers and preprocessing.
- More loss functions and metrics.
- Import/export support only where semantics can be represented correctly.

## Priority 4: Developer Experience

- Better error messages.
- More examples.
- More API reference detail.
- Reusable small-project workflow helpers that stay easy to inspect.
- Clear migration notes between releases.

## Priority 5: Scale

Large-system features such as graph execution, CUDA, distributed training, and
large model import belong after the eager CPU engine is dependable.
