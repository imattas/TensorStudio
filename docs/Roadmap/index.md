# Roadmap

TensorStudio `1.3.4` is a CPU-first tensor, autograd, neural-network, vision,
project, serialization, and ONNX-export foundation. The long-term direction is
to become a strong compact ML framework for learning, experimentation, and
lightweight workloads while staying honest about the scale of mature systems
such as PyTorch, TensorFlow, NumPy, and JAX.

This roadmap is ordered from highest-priority foundation work to later
ecosystem-scale features. Earlier phases should generally land before later
phases, because performance, correctness, and maintainability compound.

## 1. Correctness Foundation

These items should come first because every later feature depends on predictable
tensor semantics.

- Completed in `1.3.1`: define, expose, test, and document dtype promotion
  rules across binary arithmetic, division, matrix multiplication, and
  comparisons.
- Completed in `1.3.2`: add exhaustive broadcasting tests for every binary
  elementwise arithmetic and comparison operation, including functional and
  operator overload paths.
- Completed in `1.3.4`: add tuple-axis reductions for `sum`, `mean`,
  `max`, and `min`.
- Add arg reductions: `argmax` and `argmin`.
- Add `where`, `maximum`, `minimum`, `clip`, and richer comparison helpers.
- Add full NumPy-style indexing and slicing for common cases.
- Add more view/layout operations: `squeeze`, `unsqueeze`, `permute`, and
  general N-dimensional transpose.
- Add clearer shape, dtype, and indexing error messages.
- Expand numerical tests against NumPy for all new semantics.

## 2. CPU Performance Core

These items should land before broad model expansion so the framework has a
serious runtime base.

- Add BLAS-backed `matmul` for `float32` and `float64`.
- Add platform-specific BLAS selection guidance for Windows, Linux, and macOS.
- Add a small CPU thread pool for large contiguous elementwise ops, reductions,
  matrix operations, convolution, and pooling.
- Add SIMD kernels for common contiguous `float32` and `float64` elementwise
  operations.
- Improve storage allocation and reuse for temporary tensors.
- Add fast paths for scalar broadcasting and same-shape contiguous tensors.
- Add benchmark regression thresholds for core kernels.
- Keep benchmark reports honest, with win/loss columns against available local
  NumPy, PyTorch, TensorFlow, and JAX installs.

## 3. Core Math Expansion

After the tensor semantics and CPU runtime are stronger, expand the math surface
that models and scientific users expect.

- Add `logsumexp`, `softmax`, and `log_softmax` native or optimized kernels.
- Add `var`, `std`, `norm`, and common statistical reductions as first-class
  Tensor methods where appropriate.
- Add `einsum` for a practical subset of common patterns.
- Add batched matrix multiplication.
- Add random distributions beyond uniform and normal.
- Add stable numerics for cross-entropy-style workloads.
- Add `all`, `any`, and boolean reductions.

## 4. Autograd Coverage And Hardening

Autograd should grow alongside the operation set, with tests that make coverage
visible.

- Maintain a documented gradient coverage matrix.
- Add gradients for every new differentiable tensor op.
- Expand non-scalar backward coverage.
- Add graph lifecycle controls to reduce retained memory.
- Improve leaf tensor semantics and gradient accumulation behavior.
- Add safer in-place semantics for a small approved set of operations.
- Add finite-difference gradient tests for all differentiable math ops.
- Add higher-order gradient support as a later milestone.

## 5. Neural Network Building Blocks

Once math and autograd are stronger, add the modules needed for real model
families.

- Add initializers: Xavier, Kaiming, normal, uniform, zeros, and ones.
- Add `BatchNorm1d`, `BatchNorm2d`, `LayerNorm`, and `Embedding`.
- Add `Conv1d`, grouped convolution, depthwise convolution, and
  `ConvTranspose2d`.
- Add adaptive pooling and global pooling.
- Add more activation modules: GELU, ELU, SELU, SiLU, and Mish.
- Add more losses: label-smoothing cross entropy, focal loss, KL divergence,
  negative log likelihood, and cosine embedding loss.
- Add model summary utilities for parameters, shapes, and estimated memory.

## 6. Training And Project Workflows

The project utilities should evolve into a clean small-project workflow without
becoming an opaque training framework.

- Add dataset creation helpers for images, tabular arrays, labels, and
  TensorStudio tensors, with deterministic train/validation splitting and
  metadata summaries.
- Add callbacks for checkpointing, early stopping, CSV logging, and learning
  rate logging.
- Add a metrics package for regression, classification, and multilabel tasks.
- Add JSON, TOML, and YAML project config loading.
- Add checkpoint resume helpers that restore model, optimizer, scheduler, and
  epoch state.
- Add train/validation loop support in `Trainer`.
- Add deterministic seeding helpers across TensorStudio, NumPy, and Python
  random.
- Add generated project templates for classification, regression, and vision.

## 7. Computer Vision Depth

Vision should grow from classification utilities into a compact practical CV
toolkit.

- Add batched image transforms for tensor batches.
- Add color jitter, random resized crop, random rotation, affine transforms,
  cutout, mixup, and cutmix.
- Add detection utilities: NMS, box encode/decode, IoU variants, and anchor
  helpers.
- Add segmentation helpers: masks, mask IoU, and simple mask transforms.
- Add reusable model blocks: ResNet-style blocks, MobileNet-style depthwise
  blocks, and a compact UNet.
- Add dataset helpers for classification, detection, and segmentation folder
  layouts.
- Add more visualization tools for predictions, masks, and feature maps.

## 8. Serialization And Interchange

Interchange should prioritize safe, inspectable formats before broad runtime
compatibility.

- Expand NPZ metadata for richer model and optimizer state.
- Add `safetensors` support for safe tensor weight storage.
- Expand ONNX export coverage for more TensorStudio modules and ops.
- Add ONNX metadata inspection for supported files, including graph inputs,
  outputs, opset, initializer summaries, node counts, and model producer
  fields.
- Add ONNX import for a supported subset of static graphs.
- Add execution support for imported ONNX graphs only where TensorStudio has
  correct matching tensor ops; clearly reject unsupported operators and dynamic
  graph features.
- Add model metadata extraction for TensorStudio checkpoints, NPZ bundles, and
  supported ONNX files. Do not advertise metadata support for formats that
  cannot be parsed safely or accurately.
- Add import/export research for practical neural-network model formats that
  can be supported without unsafe execution or fake compatibility layers.
- Add versioned checkpoint metadata and compatibility checks.
- Add model card or metadata export for packaged examples.
- Consider DLPack-style interop after tensor/device semantics are mature.

## 9. Packaging, CI, And Release Quality

Release automation should make every published build reproducible and tested.

- Add clean wheel install tests on Windows, Linux, and macOS.
- Add clean sdist install tests where source builds are expected.
- Add manylinux, macOS universal/arm64, and Windows wheel verification.
- Add benchmark artifacts to release workflows.
- Add TestPyPI dry-run guidance before every production PyPI release.
- Prefer PyPI trusted publishing over raw tokens.
- Add docs publishing automation.
- Add ABI and platform compatibility notes for native builds.

## 10. Hardware Backends

New backends should wait until the CPU implementation has strong semantics and
test coverage.

- Add a formal device abstraction for CPU, CUDA, Metal, and future devices.
- Add CUDA storage and kernel launch infrastructure.
- Add CUDA kernels for elementwise ops, reductions, matmul, convolution, and
  pooling.
- Add Metal backend research for Apple platforms.
- Add explicit device transfer APIs.
- Add backend-specific benchmark suites.
- Add mixed precision only after dtype semantics and hardware kernels are
  stable.

## 11. Graph, Compiler, And Runtime Systems

Graph features are long-term work because they require stable eager semantics
first.

- Add tracing for a constrained subset of TensorStudio programs.
- Add graph serialization for supported traced models.
- Add basic graph optimization passes such as constant folding and op fusion.
- Add a simple JIT or ahead-of-time execution path for supported graphs.
- Add runtime profiling hooks.
- Add memory planning for graph execution.

## 12. Ecosystem And Advanced Features

These are late-stage features after the core framework is correct, fast, and
well packaged.

- Add sparse tensors.
- Add distributed training research.
- Add model zoo examples with reproducible training scripts.
- Add dataset utilities for common public formats.
- Add language-model-oriented layers and examples.
- Add quantization research.
- Add plugin extension points for custom kernels.

## Non-Goals For The Near Term

- Claiming broad superiority over mature ML frameworks.
- Production-scale distributed training.
- Full TensorFlow or PyTorch compatibility.
- CUDA-first development before the CPU backend is solid.
- Untrusted pickle loading.
- Benchmark claims that are not backed by local, reproducible results.
