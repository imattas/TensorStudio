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

