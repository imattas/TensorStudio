# Changelog

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
