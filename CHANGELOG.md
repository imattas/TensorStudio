# Changelog

## Unreleased

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
