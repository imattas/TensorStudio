# C++ First Policy

TensorStudio is primarily a C++ tensor framework with a Python API. New work
should keep performance-sensitive tensor behavior in native code whenever that
is practical and maintainable.

## Prefer C++ For

- Tensor data loops.
- Broadcasting kernels.
- Reductions.
- Autograd primitive formulas.
- Shape, dtype, and device validation.
- Serialization primitives that inspect native tensor metadata.
- CPU kernels for math, neural-network, and vision operations.

## Prefer Python For

- User-facing module composition.
- Lightweight training orchestration.
- Optional integrations with Python-native libraries such as Pillow and ONNX.
- Documentation examples.
- Tests and benchmarks.

## Review Checklist

When adding a feature, ask:

- Does this loop over tensor elements?
- Does it allocate or transform tensor storage?
- Does it need to run inside training loops?
- Would a Python implementation force repeated `tolist()` or `numpy()` copies?

If yes, implement the core in C++ and expose a thin Python wrapper.

## Honest Language Metrics

Repository language metrics should represent implementation code. Tests,
examples, generated docs, benchmark scripts, and documentation are marked for
GitHub Linguist as documentation/generated where appropriate. The Python package
layer remains visible because it is real public API code.

