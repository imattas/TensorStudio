# Platform Compatibility

TensorStudio ships a native extension module, `tensorstudio._C`, built from
C++20 through pybind11 and scikit-build-core. Install behavior depends on the
wheel tag that matches the user's Python, operating system, and CPU
architecture.

## Wheel Targets

Release workflows build CPython wheels for:

- Python `3.10`, `3.11`, `3.12`, and `3.13`.
- Windows `win_amd64`.
- Linux `manylinux_2_28_x86_64`.
- macOS host architecture plus `universal2` wheels where cibuildwheel can build
  them.

Every wheel job runs cibuildwheel's test command and then runs
`tools/verify_artifacts.py` to install the generated wheel into a clean virtual
environment and import the native extension.

## Source Distributions

The sdist is still important for platforms that do not have a matching wheel.
Source installs require:

- Python `3.10+`.
- CMake `3.18+`.
- A C++20 compiler.
- pybind11 and scikit-build-core from the isolated build environment.

The CI artifact smoke job installs the sdist in a clean environment on Windows,
Linux, and macOS. That keeps source-build failures visible before a release is
published.

## ABI Notes

TensorStudio does not use Python's limited ABI in the current release line.
Wheels are built per CPython version (`cp310`, `cp311`, `cp312`, `cp313`) rather
than as `abi3` wheels. This is the conservative choice for a pybind11 extension
whose C++ tensor core is still evolving.

The public Python API follows semantic versioning. The internal C++ ABI is not
stable yet; downstream projects should use the Python package boundary instead
of linking against TensorStudio internals.

## BLAS And CPU Feature Notes

TensorStudio can use optional BLAS-backed matrix multiplication when a compatible
provider is found at build time. Wheels must still work without BLAS, so the
portable C++ fallback remains the baseline.

CPU feature reporting is runtime metadata for benchmarks. It is not a promise
that every wheel contains hand-written kernels for every SIMD extension.

## Release Verification

Before publishing a stable release, run:

```bash
python test_all.py --quiet
python -m mkdocs build --strict
python benchmark_all.py --check-thresholds
python -m build
python -m twine check dist/*
python tools/verify_artifacts.py --wheel-dir dist --sdist-dir dist
```

On GitHub, the CI and release workflows repeat the clean install checks on each
major platform. A release should not be promoted if any platform-specific
artifact check fails.
