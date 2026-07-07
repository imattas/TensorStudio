# Roadmap

TensorStudio `1.12.0` is a CPU-first tensor, autograd, neural-network, vision,
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
- Completed in `1.3.5`: add arg reductions: `argmax` and `argmin`.
- Completed in `1.3.6`: add `where`, `maximum`, `minimum`, `clip`, and richer
  comparison helpers.
- Completed in `1.4.0`: add NumPy-style indexing and slicing for common
  integer, slice, tuple, ellipsis, newaxis, negative-index, negative-step, and
  scalar-output cases, with clear rejections for unsupported advanced indexing.
- Completed in `1.5.0`: add more view/layout operations: `squeeze`,
  `unsqueeze`, `permute`, and general N-dimensional transpose.
- Completed in `1.5.1`: add clearer shape, dtype, and indexing error
  messages.
- Expand numerical tests against NumPy for all new semantics.

## 2. CPU Performance Core

These items should land before broad model expansion so the framework has a
serious runtime base.

- Completed in `1.6.0`: add optional CBLAS/Accelerate-backed `matmul` for
  contiguous `float32` and `float64` matrices when compatible platform BLAS
  support is available, with a portable C++ fallback.
- Completed in `1.6.0`: add platform-specific BLAS selection guidance for
  Windows, Linux, and macOS.
- Completed in `1.6.0`: add a small native CPU thread pool for large
  contiguous elementwise ops, reductions, matrix operations, convolution, and
  pooling forward kernels.
- Completed in `1.6.0`: add SIMD-friendly typed `float32` and `float64`
  contiguous kernels for common elementwise arithmetic and activations.
- Completed in `1.6.0`: improve storage allocation and reuse with a bounded
  C++ storage pool.
- Completed in `1.6.0`: keep scalar broadcasting and same-shape contiguous
  tensor fast paths and make them threaded for large tensors.
- Completed in `1.6.0`: add benchmark regression thresholds for core kernels.
- Completed in `1.6.0`: keep benchmark reports honest, with win/loss columns
  against available local NumPy, PyTorch, TensorFlow, and JAX installs.

## 3. Core Math Expansion

After the tensor semantics and CPU runtime are stronger, expand the math surface
that models and scientific users expect.

- Completed in `1.7.0`: add `logsumexp`, `softmax`, and `log_softmax`
  native kernels with stable max-shifted numerics.
- Completed in `1.7.0`: add `var`, `std`, `norm`, and common statistical
  reductions as Tensor methods or first-class Python API helpers where
  appropriate.
- Completed in `1.7.0`: add `einsum` for a documented practical subset of
  common patterns.
- Completed in `1.7.0`: add batched matrix multiplication through `bmm` and
  3D `@` dispatch.
- Completed in `1.7.0`: add random distributions beyond uniform and normal,
  including `randint` and `bernoulli`.
- Completed in `1.7.0`: add stable numerics for cross-entropy-style workloads.
- Completed in `1.7.0`: add `all`, `any`, and boolean reductions.

## 4. Autograd Coverage And Hardening

Autograd should grow alongside the operation set, with tests that make coverage
visible.

- Completed in `1.8.0`: maintain a documented gradient coverage matrix.
- Completed in `1.8.0`: add gradients for every new differentiable tensor op
  from the `1.7.0` math expansion.
- Completed in `1.8.0`: expand non-scalar backward coverage with explicit
  gradient tests through newer differentiable ops.
- Completed in `1.8.0`: add graph lifecycle controls to reduce retained memory,
  including retained-graph backward and normal graph release.
- Completed in `1.8.0`: improve leaf tensor semantics and gradient
  accumulation behavior.
- Completed in `1.8.0`: add safer in-place semantics for a small approved set
  of operations.
- Completed in `1.8.0`: add finite-difference gradient tests for the current
  differentiable math surface.
- Deferred beyond `1.8.0`: higher-order gradient support remains a later
  engine-design milestone because the current backward engine intentionally
  detaches backward gradients.

## 5. Neural Network Building Blocks

Once math and autograd are stronger, add the modules needed for real model
families.

- Completed in `1.9.0`: add initializers: Xavier, Kaiming, normal, uniform,
  zeros, and ones.
- Completed in `1.9.0`: add `BatchNorm1d`, `BatchNorm2d`, `LayerNorm`, and
  `Embedding`.
- Completed in `1.9.0`: add `Conv1d`, grouped convolution, depthwise
  convolution, and
  `ConvTranspose2d`.
- Completed in `1.9.0`: add adaptive pooling and global pooling.
- Completed in `1.9.0`: add more activation modules: GELU, ELU, SELU, SiLU,
  and Mish.
- Completed in `1.9.0`: add more losses: label-smoothing cross entropy,
  focal loss, KL divergence, negative log likelihood, and cosine embedding
  loss.
- Completed in `1.9.0`: add model summary utilities for parameters, shapes,
  and estimated memory.

## 6. Training And Project Workflows

The project utilities should evolve into a clean small-project workflow without
becoming an opaque training framework.

- Completed in `1.10.0`: add dataset creation helpers for images, tabular arrays, labels, and
  TensorStudio tensors, with deterministic train/validation splitting and
  metadata summaries.
- Completed in `1.10.0`: add callbacks for checkpointing, early stopping, CSV logging, and learning
  rate logging.
- Completed in `1.10.0`: add a metrics package for regression, classification, and multilabel tasks.
- Completed in `1.10.0`: add JSON, TOML, and YAML project config loading.
- Completed in `1.10.0`: add checkpoint resume helpers that restore model, optimizer, scheduler, and
  epoch state.
- Completed in `1.10.0`: add train/validation loop support in `Trainer`.
- Completed in `1.10.0`: add deterministic seeding helpers across TensorStudio, NumPy, and Python
  random.
- Completed in `1.10.0`: add generated project templates for classification, regression, and vision.

## 7. Computer Vision Depth

Vision should grow from classification utilities into a compact practical CV
toolkit.

- Completed in `1.11.0`: add batched image transforms for tensor batches.
- Completed in `1.11.0`: add color jitter, random resized crop, random rotation, affine transforms,
  cutout, mixup, and cutmix.
- Completed in `1.11.0`: add detection utilities: NMS, box encode/decode, IoU variants, and anchor
  helpers.
- Completed in `1.11.0`: add segmentation helpers: masks, mask IoU, and simple mask transforms.
- Completed in `1.11.0`: add reusable model blocks: ResNet-style blocks, MobileNet-style depthwise
  blocks, and a compact UNet.
- Completed in `1.11.0`: add dataset helpers for classification, detection, and segmentation folder
  layouts.
- Completed in `1.11.0`: add more visualization tools for predictions, masks, and feature maps.

## 8. Serialization And Interchange

Interchange should prioritize safe, inspectable formats before broad runtime
compatibility.

- Completed in `1.12.0`: expand NPZ metadata for richer model and optimizer state.
- Completed in `1.12.0`: add `safetensors` support for safe tensor weight storage.
- Completed in `1.12.0`: expand ONNX export coverage for more TensorStudio modules and ops.
- Completed in `1.12.0`: add ONNX metadata inspection for supported files, including graph inputs,
  outputs, opset, initializer summaries, node counts, and model producer
  fields.
- Completed in `1.12.0`: add ONNX import for a supported subset of static graphs.
- Completed in `1.12.0`: add execution support for imported ONNX graphs only where TensorStudio has
  correct matching tensor ops; clearly reject unsupported operators and dynamic
  graph features.
- Completed in `1.12.0`: add model metadata extraction for TensorStudio checkpoints, NPZ bundles, and
  supported ONNX files. Do not advertise metadata support for formats that
  cannot be parsed safely or accurately.
- Completed in `1.12.0`: add import/export research for practical neural-network model formats that
  can be supported without unsafe execution or fake compatibility layers.
- Completed in `1.12.0`: add versioned checkpoint metadata and compatibility checks.
- Completed in `1.12.0`: add model card or metadata export for packaged examples.
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
