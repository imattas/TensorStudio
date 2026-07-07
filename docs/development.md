# Development

This guide is for contributors changing the C++ core, Python API, tests, docs,
examples, benchmarks, packaging, or CI.

## Repository Layout

- `include/tensorstudio`: public C++ headers.
- `src/core`: tensor, storage, ops, autograd, dtype, shape, device, random, and
  utility implementations.
- `src/bindings`: pybind11 bindings for `tensorstudio._C`.
- `python/tensorstudio`: Python public API, nn, optim, data, serialization, and
  typing.
- `tests`: pytest suite.
- `examples`: runnable examples.
- `benchmarks`: local timing scripts.
- `docs`: MkDocs documentation.
- `.github/workflows`: CI, wheels, and publish workflows.

## Build Requirements

TensorStudio builds a C++20 extension module named `tensorstudio._C`.

Windows source builds require MSVC. Linux source builds require GCC or Clang.
macOS source builds require Apple Clang through Xcode Command Line Tools.

End users should normally install wheels. Wheels should not require CMake or a
compiler at install time.

## Development Install

```bash
python -m pip install -U pip
python -m pip install -e ".[dev,docs]"
```

## Common Checks

```bash
ruff check .
mypy python/tensorstudio
pytest -q
python test_all.py --skip-build
python -m build
python -m twine check dist/*
```

Run examples:

```bash
python examples/basic_tensor_ops.py
python examples/linear_regression.py
python examples/tiny_mlp.py
python examples/save_load_model.py
```

## Windows-First Checklist

```powershell
python -m pip install -U pip
python -m pip install -e ".[dev]"
python -c "import tensorstudio as ts; import tensorstudio._C; print(ts.__version__); print(ts.ones((2, 2)) + 1)"
pytest -q
python -m build
python -m twine check dist/*
```

Clean wheel and sdist tests are documented in [Windows](windows.md).

## Documentation

```bash
mkdocs serve
mkdocs build
```

Docs should describe implemented behavior and limitations. Do not claim
TensorStudio is better or faster than mature ML frameworks.

## C++ Guidelines

- Prefer readable loops and shape helpers over template-heavy kernels.
- Use RAII and standard library containers.
- Avoid raw owning pointers.
- Keep dtype, shape, device, and autograd errors explicit.
- Keep C++ portable across MSVC, GCC, Clang, and Apple Clang.
- Avoid POSIX-only assumptions in core paths.
- Add gradient formulas close to the forward op implementation where practical.

## Python Guidelines

- Keep wrappers thin over the C++ core.
- Type public functions.
- Maintain accurate `__all__` exports.
- Avoid circular imports.
- Do not use `eval` or `exec`.
- Do not make network calls from library code.
- Treat pickle loading as trusted-input only.

## Testing Strategy

Tests should cover:

- Import and version.
- Tensor creation, dtype handling, shape metadata, and views.
- Broadcasting and broadcasting gradients.
- Forward results against NumPy.
- Autograd analytical and finite-difference checks.
- Neural network modules and losses.
- Optimizer updates and state dictionaries.
- Serialization roundtrips.
- DataLoader batching and deterministic shuffle.
- NumPy interop copies.

Use `np.testing.assert_allclose` for numerical comparisons.

## Release Hygiene

For release work, keep documentation and benchmark claims aligned with measured
results. A stable API release is not the same thing as a performance-superiority
claim.

`python test_all.py` runs the local release gate. Use `--skip-build` or
`--skip-docs` while iterating, then run it without skips before a release.

`python benchmark_all.py` regenerates `benchmarks/results.md` with all benchmark
sections and explicit per-framework win columns.
