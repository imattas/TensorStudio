# Build From Source

Use a source build when developing TensorStudio or testing a platform wheel
fallback.

## Editable Development Build

```bash
python -m pip install -U pip
python -m pip install -e ".[dev,docs]"
```

This builds `tensorstudio._C` through scikit-build-core and CMake.

## Local Release Build

```bash
python -m build
python -m twine check dist/*
```

The wheel should contain the compiled native extension inside the
`tensorstudio` package directory.

## Backend Descriptor Flags

TensorStudio's default build is CPU-only. Two CMake flags can mark future
backend descriptor hooks as compiled for diagnostics, without enabling tensor
storage or kernel execution on those devices:

```bash
python -m pip install -e ".[dev]" \
  -Ccmake.define.TENSORSTUDIO_ENABLE_CUDA_BACKEND_DESCRIPTOR=ON
python -m pip install -e ".[dev]" \
  -Ccmake.define.TENSORSTUDIO_ENABLE_METAL_BACKEND_DESCRIPTOR=ON
```

These flags are for backend-boundary experiments only. They do not make CUDA
or Metal tensors usable.

## Platform Compilers

Windows:

- Install Visual Studio Build Tools with the C++ workload.
- Use a Developer PowerShell or ensure MSVC is discoverable by CMake.

Linux:

- GCC or Clang with C++20 support is required.
- Manylinux wheels are built in CI; local source builds use your host compiler.

macOS:

- Install Xcode Command Line Tools.
- Apple Clang is the expected compiler.

## Full Local Gate

```bash
python test_all.py --quiet
```

This runs linting, type checks, tests, examples, docs build, package build, and
Twine metadata validation.
