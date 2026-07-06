# Development

This guide is for contributors changing C++ core code, Python wrappers, tests,
docs, or packaging.

## Repository Layout

- `include/tensorstudio`: public C++ headers
- `src/core`: C++ tensor, storage, ops, autograd, dtype, shape, and utilities
- `src/bindings`: pybind11 extension bindings
- `python/tensorstudio`: Python API, modules, optimizers, data helpers, typing
- `tests`: pytest suite
- `examples`: runnable examples
- `benchmarks`: local timing scripts
- `docs`: MkDocs documentation
- `.github/workflows`: CI, wheel build, and publish automation

## Native Build Requirements

TensorStudio builds a C++20 extension module named `tensorstudio._C`.

Windows:

```powershell
# Use a Visual Studio Developer PowerShell, or install wheels from PyPI.
python -m pip install -e ".[dev]"
```

macOS:

```bash
xcode-select --install
python -m pip install -e ".[dev]"
```

Linux:

```bash
sudo apt-get install build-essential cmake
python -m pip install -e ".[dev]"
```

End users should normally install prebuilt wheels from PyPI and should not need
CMake or a compiler.

## Development Install

```bash
python -m pip install -U pip
python -m pip install -e ".[dev,docs]"
```

## Common Checks

```bash
ruff check .
mypy python/tensorstudio
pytest
python -m build
```

## Documentation

Serve docs locally:

```bash
mkdocs serve
```

Build static docs:

```bash
mkdocs build
```

Docs should explain limitations and failure modes. Do not describe TensorStudio
as superior to mature ML frameworks.

## C++ Guidelines

- Prefer clear loops and shape helpers over dense template metaprogramming.
- Use RAII and standard library containers.
- Avoid raw owning pointers.
- Keep dtype and shape errors explicit.
- Add gradient formulas next to the forward op implementation when practical.
- Keep the CPU backend portable across Linux, macOS, and Windows.

## Python Guidelines

- Keep wrappers thin.
- Keep public APIs typed.
- Use `__all__` exports.
- Avoid runtime network calls in library code.
- Do not use `eval` or `exec`.
- Treat pickle loading as trusted-input only.

## Testing Strategy

Add tests for tensor creation, dtype handling, broadcasting, shape errors, NumPy
comparisons, autograd gradients, modules, optimizers, and serialization.

For numerical tests, prefer `np.testing.assert_allclose`.

## Release Hygiene

Before a release:

```bash
ruff check .
mypy python/tensorstudio
pytest
python -m build
twine check dist/*
```

Wheels are built in CI with cibuildwheel so users can install without a local
C++ toolchain.
