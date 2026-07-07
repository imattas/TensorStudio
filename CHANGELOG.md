# Changelog

## Unreleased

## 1.11.0 - 2026-07-07

- Completed the ordered Computer Vision Depth roadmap section as one release
  batch.
- Added batch-aware resize/crop/normalize helpers plus color jitter, random
  resized crop, random rotation, affine transforms, cutout, mixup, and CutMix.
- Added detection utilities for box areas, IoU variants, NMS, box
  encode/decode, coordinate conversion, and anchor generation.
- Added segmentation helpers for mask IoU, one-hot conversion, masks-to-boxes,
  nearest mask resize, and deterministic mask crops.
- Added `DetectionFolder` and `SegmentationFolder` datasets plus
  `tensorstudio.data` factory aliases for detection and segmentation folders.
- Added `ResidualBlock`, `DepthwiseSeparableBlock`, `CompactUNet`, and
  `make_unet()` vision model helpers using the native-backed neural-network
  layer stack.
- Added prediction drawing, mask overlay, and feature-map grid visualization
  helpers.
- Expanded vision tests and documentation for transforms, detection,
  segmentation, model blocks, and visualization.

## 1.10.0 - 2026-07-07

- Completed the ordered Training And Project Workflows roadmap section as one
  release batch.
- Added `ArrayDataset`, tensor/array/image-folder dataset factories,
  deterministic train/validation splitting, and dataset metadata summaries.
- Added `tensorstudio.metrics` with regression, classification, and multilabel
  metrics for small supervised workflows.
- Added trainer validation loops, scheduler stepping, callback context support,
  learning-rate logging, CSV logging, checkpoint callbacks, and early stopping.
- Added JSON, TOML, and YAML project config loading plus deterministic seeding
  across TensorStudio, NumPy, and Python random.
- Expanded full checkpoints with scheduler and epoch state and added
  `resume_checkpoint()` for continuing training runs.
- Added generated regression, classification, and vision project templates.
- Expanded tests and docs for project workflows, metrics, dataset creation,
  callbacks, configs, templates, and checkpoint resume.

## 1.9.0 - 2026-07-07

- Completed the ordered Neural Network Building Blocks roadmap section as one
  release batch.
- Added native grouped `conv2d`, native `conv_transpose2d`, and native
  embedding lookup with autograd support.
- Added Python-level `Conv1d`, `DepthwiseConv2d`, `ConvTranspose2d`,
  `BatchNorm1d`, `BatchNorm2d`, `LayerNorm`, `Embedding`,
  adaptive/global pooling, and additional activation modules.
- Added `tensorstudio.nn.init` with Xavier, Kaiming, normal, uniform, zero, and
  one initializers.
- Added label-smoothing cross entropy, focal loss, KL divergence, negative log
  likelihood, and cosine embedding loss modules and functional helpers.
- Added module buffers, buffer-aware `state_dict()` support, and model summary
  utilities for parameters, shapes, and estimated tensor memory.
- Expanded tests and docs for the section-5 neural-network API surface.

## 1.8.0 - 2026-07-07

- Completed the ordered Autograd Coverage And Hardening roadmap section as one
  release batch.
- Added `retain_graph` support to `Tensor.backward()` and
  `tensorstudio.autograd.backward()`.
- Added graph lifecycle hardening: normal backward frees non-leaf graph history,
  repeated backward through a freed graph raises a clear error, and retained
  graphs clear intermediate gradients between backward passes.
- Added Tensor `is_leaf`, `clear_history()`, and `detach_()` controls for
  explicit graph lifecycle management.
- Added guarded public in-place methods `zero_()`, `fill_()`, and `add_()`.
  They reject gradient-tracked mutation while grad mode is enabled and work
  inside `tensorstudio.no_grad()`.
- Expanded non-scalar backward and finite-difference gradient tests for stable
  probability ops, statistics, norms, and batched matrix multiplication.
- Expanded autograd documentation with a coverage matrix, lifecycle notes, and
  explicit higher-order-gradient limitations.

## 1.7.0 - 2026-07-07

- Completed the ordered Core Math Expansion roadmap section as one release
  batch.
- Added native C++ `logsumexp`, `softmax`, and `log_softmax` operations with
  max-shifted stable numerics and autograd support.
- Added native C++ batched matrix multiplication through `bmm` and 3D `@`
  dispatch, including reverse-mode gradients for both operands.
- Added native C++ `var`, `variance`, `std`, `all`, and `any` operations, plus
  Tensor methods and top-level Python exports where appropriate.
- Added Tensor-level `norm()` and expanded `tensorstudio.math` with
  `logsumexp`, `softmax`, `log_softmax`, boolean reductions, and a documented
  practical `einsum` subset.
- Added seeded native random distributions: `uniform`, `normal`, `randint`,
  and `bernoulli`, with Python `*_like` helpers where useful.
- Switched neural-network functional softmax/log-softmax and cross entropy to
  the native stable kernels.
- Added NumPy parity and autograd tests for the expanded math surface.

## 1.6.0 - 2026-07-07

- Completed the ordered CPU Performance Core roadmap section as one release
  batch.
- Added optional CBLAS/Accelerate-backed `matmul` for contiguous `float32` and
  `float64` matrices when the source build environment exposes a compatible
  BLAS library and header; portable C++ kernels remain the fallback path.
- Added a small native C++ CPU thread pool, configurable with
  `tensorstudio.set_num_threads()` and `TENSORSTUDIO_NUM_THREADS`, and used it
  for large contiguous elementwise ops, reductions, matrix multiplication,
  convolution, and pooling forward kernels.
- Added SIMD-friendly typed `float32`/`float64` contiguous kernels for common
  elementwise arithmetic and activations, while preserving mixed-dtype fallback
  behavior.
- Added a bounded C++ storage reuse pool for tensor allocations, with
  `TENSORSTUDIO_DISABLE_STORAGE_POOL=1` and
  `TENSORSTUDIO_STORAGE_POOL_MAX_BLOCK_BYTES` controls.
- Added `tensorstudio.performance_info()`, `get_num_threads()`, and
  `set_num_threads()` diagnostics/configuration helpers.
- Added benchmark threshold support via `benchmark_all.py --check-thresholds`
  and `benchmarks/thresholds.json`.
- Expanded performance, CPU backend, and platform docs for BLAS selection,
  threading, storage reuse, benchmark thresholds, and honest interpretation.

## 1.5.1 - 2026-07-07

- Completed the next ordered correctness-roadmap item with clearer native
  shape, dtype, and indexing error messages.
- Broadcasting errors now include the mismatched axis and dimensions.
- Reshape errors now include requested element counts and inferred-shape
  context.
- DType errors now validate Python dtype inputs and list supported dtype names
  and aliases.
- Indexing errors now include tensor shape/rank context and the unsupported
  Python index type/value where practical.
- Added focused regression tests for the improved diagnostics.

## 1.5.0 - 2026-07-07

- Completed the next ordered correctness-roadmap item with native C++ view and
  layout operations: `squeeze`, `unsqueeze`, `permute`, and general
  N-dimensional transpose.
- Added Python Tensor methods, functional APIs, top-level exports, and type
  stubs for the new layout operations.
- Added autograd support for the new metadata views, including non-contiguous
  gradient handling through reshape chains.
- Added NumPy parity tests for view semantics, strides, shape errors, and
  autograd behavior.
- Updated tensor, API, autograd, and roadmap docs for the expanded view API.

## 1.4.0 - 2026-07-07

- Supersedes the short-lived `1.3.7` indexing release as a minor version
  because the indexing/view API is a larger public surface change than a patch.
- Completed the next ordered correctness-roadmap item with native C++ basic
  indexing and slicing views.
- Added Python `Tensor.__getitem__` support for integers, slices with steps,
  tuples, ellipsis, and `None`/newaxis while rejecting advanced list, tensor,
  and boolean indexing clearly.
- Added autograd scatter-back support for differentiable integer/slice views.
- Added NumPy parity tests for common indexing and slicing cases, scalar
  indexing, negative strides, empty slices, unsupported indexing errors, and
  indexing autograd.
- Expanded docs into multi-page sections for sparse documentation folders:
  data, performance, project workflows, release process, and roadmap.

## 1.3.6 - 2026-07-07

- Completed the next ordered correctness-roadmap item with native C++
  `where`, `maximum`, and `minimum` operations.
- Exposed top-level comparison helpers, `clamp`/`clip`, selection helpers, and
  Tensor method helpers through the Python API while keeping kernels in C++.
- Added NumPy parity and autograd tests for `where`, `maximum`, and `minimum`,
  including broadcasting and tie-gradient behavior.
- Reorganized the MkDocs tree into categorized sections and added deeper docs
  for installation, source builds, native C++ architecture, broadcasting,
  gradient behavior, model formats, dataset creation, and the C++-first policy.
- Added GitHub Linguist attributes so language statistics emphasize the native
  C++ library and do not count docs, examples, tests, generated site output, or
  benchmark scripts as implementation code.

## 1.3.5 - 2026-07-07

- Completed the next ordered correctness-roadmap item with native C++
  `argmax` and `argmin` operations.
- Added Tensor methods, functional ops, top-level Python exports, and type
  stubs for arg reductions.
- Added NumPy parity tests for all-element and axis arg reductions, including
  `keepdims`, negative axes, first-tie behavior, dtype, and error cases.
- Documented arg-reduction semantics and limitations.

## 1.3.4 - 2026-07-07

- Completed the next ordered correctness-roadmap item with tuple/list-axis
  reductions for `sum`, `mean`, `max`, and `min`.
- Added C++ binding-layer axis normalization for reduction methods and
  functional ops, including negative-axis handling, duplicate-axis errors, and
  `axis=()` no-op behavior.
- Added NumPy parity tests and autograd coverage for tuple-axis reductions.
- Added ordered roadmap entries for supported ONNX/model metadata extraction,
  supported ONNX import/execution, safe model-format research, and image/tensor
  dataset creation helpers.

## 1.3.3 - 2026-07-07

- Hardened CI type checks for newer Python/mypy combinations by annotating
  NumPy temporaries in vision transform and visualization helpers.
- This patch does not change runtime behavior.

## 1.3.2 - 2026-07-07

- Completed the second ordered roadmap item with exhaustive binary
  elementwise broadcasting tests.
- Added NumPy comparison coverage for broadcasted `add`, `sub`, `mul`, `div`,
  equality, inequality, and ordered comparisons.
- Added incompatible-broadcast error tests and operator-overload broadcasting
  parity tests.

## 1.3.1 - 2026-07-07

- Started the ordered roadmap release train with the dtype-promotion contract.
- Exposed native `promote_types` plus Python `tensorstudio.dtypes`,
  `promote_types`, `result_type`, `normalize_dtype`, and `dtype_of` helpers.
- Documented arithmetic, division, and comparison dtype result rules.
- Added tests that pin binary operation result dtypes across supported dtypes.

## 1.3.0 - 2026-07-07

- Expanded `tensorstudio.vision` into a broader computer-vision toolkit with
  image IO, transform pipelines, deterministic augmentation helpers, image
  grids, and bounding-box drawing.
- Added `ImageFolder` and `ImageList` datasets for local image classification
  workflows.
- Added classification metrics (`accuracy`, `top_k_accuracy`,
  `confusion_matrix`) and detection utility `box_iou`.
- Added reusable vision modules `ConvBlock`, `ImageClassifier`, and
  `make_image_classifier`.
- Updated ONNX export flattening so nested vision model blocks export through
  their supported TensorStudio layers.
- Added an ImageFolder classification example and expanded vision tests.
- Added native C++ advanced math ops (`sin`, `cos`, `tan`, `asin`, `acos`,
  `atan`, `log1p`, `rsqrt`) with autograd coverage.
- Added `tensorstudio.math` helpers for square, reciprocal, variance, standard
  deviation, and norms.
- Added `tensorstudio.project` with JSON project configs, run folders, a small
  eager `Trainer`, safe NPZ state-dict checkpoints, and trusted full
  checkpoints.
- Added a project-training example and project workflow tests.

## 1.2.0 - 2026-07-07

- Added `save_npz` and `load_npz` for non-pickle tensor and flat
  `state_dict` files using NumPy NPZ archives plus TensorStudio JSON metadata.
- Added optional ONNX export through `tensorstudio.export_onnx` and
  `tensorstudio.interchange.export_onnx` for supported Sequential module
  stacks.
- Added `tensorstudio.vision` with image-to-tensor conversion, normalization,
  center crop, nearest-neighbor resize, and a small `TinyConvClassifier`.
- Added tests for NPZ roundtrips, ONNX graph validation, and vision classifier
  forward/backward behavior.
- Updated README and MkDocs pages for file formats, ONNX limits, and vision
  workflows.

## 1.1.0 - 2026-07-07

- Added native CPU NCHW `conv2d` with stride, padding, dilation, optional bias,
  and reverse-mode gradients for input, weight, and bias.
- Added `tensorstudio.conv2d`, `tensorstudio.ops.conv2d`,
  `tensorstudio.nn.functional.conv2d`, and `tensorstudio.nn.Conv2d`.
- Added convolution tests, finite-difference autograd coverage, and a
  standalone convolution benchmark script.
- Added convolution rows to the full benchmark report and updated docs to mark
  convolution as partially implemented rather than missing.
- Added native CPU NCHW `max_pool2d` and `avg_pool2d` operations with autograd,
  Python top-level and functional APIs, `nn.MaxPool2d`, `nn.AvgPool2d`, tests,
  docs, and benchmark coverage.
- Added native single-axis `sum`, `mean`, `max`, and `min` reductions with
  `keepdims` and reverse-mode gradients.
- Added a typed contiguous 2D fast path for common axis `sum`/`mean`
  reductions.
- Added `nn.softmax`, `nn.log_softmax`, `nn.cross_entropy`, and
  `nn.CrossEntropyLoss` for small multiclass classification workloads.
- Added native `astype`, `concat`, and `stack` operations with Python wrappers,
  Tensor methods for casting, and autograd support for floating tensors.

## 1.0.1

- Added `test_all.py` for one-command local release checks covering lint,
  typing, tests, examples, docs, build, and package metadata validation.
- Added `benchmark_all.py` for one-command full benchmark report generation.
- Added explicit benchmark win columns for NumPy, TensorFlow, PyTorch, and JAX,
  plus fastest-library reporting for each benchmark case.
- Hardened release publishing triggers so future `v*` tags are eligible for
  the publish workflow instead of only `v1.0.0`.
- Updated docs to describe benchmark interpretation, local release checks, and
  site navigation for benchmark, hardware, roadmap, and usage pages.

## 1.0.0

- Promoted the CPU-only stable API foundation from release candidate status.
- Added direct C++ buffer copies for NumPy import/export paths.
- Added faster typed tensor construction, flat-vector export, contiguous tensor
  copy, and floating-point optimizer update paths.
- Added a `float32` contiguous matmul fast path that avoids double accumulation
  and per-element dtype conversion for the common training dtype.
- Added typed contiguous reduction kernels for `sum` and `mean`.
- Expanded benchmark reports with framework version metadata and PyTorch
  win/loss summaries.
- Updated documentation to describe honest `1.0.0` performance boundaries.

## 1.0.0rc2

- Added `*_like` tensor creation helpers for zero, one, full, empty, uniform
  random, and normal random tensors.
- Added `nn.Identity`, `nn.LeakyReLU`, `nn.Softplus`,
  `nn.BCEWithLogitsLoss`, `nn.HuberLoss`, and matching functional helpers.
- Expanded `nn.Module` with child/module introspection, trainable parameter
  filtering, freeze/unfreeze helpers, parameter counting, recursive `apply`,
  and richer `repr` details.
- Added optimizer utilities for gradient clipping and lightweight
  `StepLR`/`ExponentialLR` learning-rate schedulers.
- Added `len(DataLoader)` batch counting with `drop_last` support.
- Added typed C++ fast paths for contiguous elementwise/unary kernels and a
  contiguous matmul kernel that avoids per-element Tensor method dispatch.
- Bumped the next release candidate after public `1.0.0rc1` artifacts.

## 1.0.0rc1

- Release-candidate hardening toward a CPU-only v1.0.0 API foundation.
- Added additional tensor creation helpers, math operations, comparisons,
  no-grad mode, state APIs, data utilities, and Windows-first CI/release docs.
- This is not final v1.0.0; remaining platform checklist items must pass before
  a stable release.

## 0.1.1

- Publishable wheel release for CPython 3.10-3.13.
- Fixed MSVC build portability by removing non-portable `ssize_t` usage.
- Updated wheel CI to build 64-bit Windows wheels and modern Linux wheels.
- Expanded documentation with MkDocs navigation and detailed usage guides.

## 0.1.0

- Initial experimental release.
- C++20 CPU tensor core with Python bindings.
- Basic broadcasting, matrix multiplication, reductions, activations, and views.
- Reverse-mode autograd for the initial operation set.
- Python `nn`, `optim`, serialization, examples, tests, docs, and CI workflows.
