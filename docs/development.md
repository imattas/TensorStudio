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
# Use a Visual Studio Developer PowerShell, or make sure cl/nmake are on PATH.
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

Add tests for:

- Tensor creation and dtype behavior
- Broadcasting and shape errors
- NumPy numerical comparisons
- Autograd gradients for new differentiable ops
- Python module or optimizer behavior
- Serialization roundtrips

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

The wheel build requires a working native compiler. If the local machine cannot
compile the extension, use CI wheel builds.
