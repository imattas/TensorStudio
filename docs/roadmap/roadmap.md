# Roadmap

TensorStudio `2.0.0` is a CPU-first tensor, autograd, neural-network, vision,
project, serialization, ONNX, graph-runtime, and lightweight ecosystem
foundation. This page lists only remaining or intentionally deferred work.
Completed release history belongs in `CHANGELOG.md` and
`docs/roadmap/milestones.md`.

The roadmap stays ordered from core correctness and runtime work to broader
ecosystem features. TensorStudio should keep improving performance and coverage
without claiming broad superiority over mature systems such as PyTorch,
TensorFlow, NumPy, and JAX unless reproducible benchmarks prove a specific
claim.

## 1. Correctness And Autograd Depth

- Expand NumPy-comparison tests for every newly added tensor semantic,
  especially edge cases around dtype promotion, indexing, reshaping, and
  reductions.
- Add full advanced indexing support for list, tensor, and boolean-mask
  indexing, including clear gradient behavior where differentiable.
- Add a richer dtype casting policy with explicit safe, same-kind, and unsafe
  conversion modes.
- Add higher-order gradients. The current backward engine intentionally
  detaches backward gradients.
- Add broader in-place autograd tracking beyond the current small approved
  mutation set.
- Add gradient checkpointing for memory-heavy eager models.
- Keep the public gradient coverage matrix synchronized with every new
  differentiable operation.

## 2. CPU Performance Core

- Add runtime-dispatched SIMD kernels beyond compiler autovectorization.
- Improve non-BLAS matrix multiplication with blocking, cache-aware tiling, and
  better small-matrix fast paths.
- Ship clearer BLAS provider options for Windows wheels and local source builds.
- Add more threaded backward kernels for reductions, matrix multiplication,
  convolution, pooling, and normalization.
- Continue improving storage reuse with allocator telemetry and stricter
  regression tests.
- Add benchmark thresholds for new ecosystem surfaces such as sparse operations,
  language helpers, and ONNX import/runtime paths.
- Keep benchmark reports honest with win/loss columns against locally available
  NumPy, PyTorch, TensorFlow, and JAX installs.

## 3. Hardware Backends

- Implement real CUDA tensor storage and execution kernels. CUDA support should
  include elementwise ops, reductions, matrix multiplication, convolution,
  pooling, and autograd kernels before it is advertised as usable.
- Add CUDA CI and wheel-build coverage before publishing CUDA-capable releases.
- Implement a Metal/MPS execution backend for Apple platforms after the backend
  boundary is proven against CUDA and CPU.
- Add explicit device-to-device copy semantics for every supported backend pair.
- Add mixed precision only after dtype semantics and hardware kernels are
  stable.
- Add backend-specific benchmark suites that compare CPU, CUDA, Metal, and any
  future devices without hiding losses.

## 4. Graph, Compiler, And Runtime Systems

- Expand tracing coverage beyond the current constrained tensor-program subset.
- Add explicit graph diagnostics for unsupported Python control flow and dynamic
  shapes.
- Add stronger graph optimization passes: common subexpression elimination,
  dead-node elimination, shape propagation, and broader fusion.
- Add fused eager execution paths for common graph patterns.
- Research machine-code JIT or ahead-of-time native code generation once eager
  semantics and kernel coverage are stable.
- Add graph-level autograd support for compiled/traced programs.
- Add persistent graph profiling artifacts that can be compared across releases.

## 5. Distributed And Large Training

- Add real multi-process collectives with a tested transport backend.
- Add distributed data parallel training with deterministic parameter
  synchronization and failure handling.
- Add distributed checkpoint save/load helpers.
- Add rank-aware logging, metrics, and benchmark reporting.
- Add CI coverage for at least a local multi-process distributed smoke test.
- Keep production-scale distributed training out of the public claims until the
  runtime has real transport, recovery, and scaling tests.

## 6. Serialization, Interchange, And Model Formats

- Expand TensorStudio ONNX import/export coverage for more operators and module
  families.
- Add TensorStudio-to-DLPack export and non-CPU DLPack ownership after device
  storage semantics are mature.
- Research safe metadata inspection for TensorFlow SavedModel, Keras `.keras`,
  and HDF5 weight files without executing untrusted model code.
- Add tested tensor-only HDF5 import/export only if a stable schema is defined.
- Add versioned compatibility checks for every non-pickle storage format.

## 7. Data, Vision, And Model Coverage

- Add larger model-zoo examples with reproducible training scripts and metadata.
- Add pretrained-weight metadata only where weights are legally distributable
  and checksummed.
- Add end-to-end detection and segmentation training examples.
- Add video IO helpers after the image path is stable.
- Add more vision models, including residual, mobile, UNet-style, and simple
  transformer-backed examples.
- Add broader transformer stacks and sequence-model examples.
- Add dataset download/caching workflows with integrity checks for approved
  public datasets.

## 8. Sparse, Quantization, And Kernel Extensions

- Expand sparse tensor formats beyond COO/CSR, such as CSC where useful and
  testable.
- Add sparse autograd coverage for supported sparse-dense operations.
- Add quantized inference kernels instead of only affine quantization helpers
  and fake quantization.
- Define a stable native plugin ABI for custom C++ kernels.
- Add safe plugin discovery that does not execute arbitrary untrusted code.

## 9. Packaging, CI, And Release Quality

- Keep publishing through trusted GitHub/PyPI release paths where possible.
- Add release verification for every wheel artifact built on GitHub Actions.
- Add clean install tests for optional extras such as `onnxruntime`,
  `safetensors`, and `vision`.
- Add ABI compatibility notes for each native-extension release.
- Keep TestPyPI or equivalent dry-run guidance current.
- Keep docs, examples, benchmark reports, and README claims synchronized with
  the released package.

## Non-Goals For The Near Term

- Claiming broad superiority over mature ML frameworks.
- Full TensorFlow or PyTorch compatibility.
- Production-scale distributed training before real transport backends exist.
- CUDA-first development before CPU semantics and tests remain stable.
- Loading untrusted pickle files.
- Loading arbitrary neural-network formats by executing untrusted code.
- Benchmark claims that are not backed by local, reproducible results.
