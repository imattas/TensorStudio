# Windows

Windows is TensorStudio's first release target for v1.
The goal is for normal users to install wheels without CMake and for
contributors to build from source with MSVC.

## Wheel Install

```powershell
python -m pip install -U pip
python -m pip install tensorstudio
python -c "import tensorstudio as ts; import tensorstudio._C; print(ts.__version__)"
```

If a wheel is available for your Python version, pip should not invoke CMake.

## Source Build Requirements

Install:

- Python 3.10 or newer from python.org.
- Microsoft C++ Build Tools or Visual Studio.
- The Desktop development with C++ workload.
- CMake, if not provided through the Python build environment.

Then run from PowerShell:

```powershell
python -m pip install -U pip
python -m pip install -e ".[dev]"
python -c "import tensorstudio as ts; import tensorstudio._C; print(ts.__version__)"
pytest -q
```

If `nmake`, `cl`, or MSVC tools are missing, source builds will fail. That is a
toolchain issue, not a TensorStudio runtime requirement for wheel users.

## Optional BLAS

TensorStudio does not require BLAS to build or run. For source builds that want
the optional CBLAS-backed `float32` and `float64` matrix multiplication path,
install a CBLAS-compatible provider such as OpenBLAS or MKL and make both the
library and `cblas.h` visible to CMake.

Common Windows routes:

- vcpkg with OpenBLAS or another CBLAS-compatible package.
- Intel oneAPI MKL with environment variables initialized before running pip.
- A system OpenBLAS install on `PATH` plus include/library paths exposed to
  CMake.

Verify the resulting build with:

```powershell
python -c "import tensorstudio as ts; print(ts.performance_info())"
```

If `blas_enabled` is `False`, TensorStudio is using the portable C++ fallback.

## Build And Check

```powershell
python -m build
python -m twine check dist/*
```

## Clean Wheel Test

```powershell
python -m venv .venv-wheel-test
.\.venv-wheel-test\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install dist/*.whl
python -c "import tensorstudio as ts; import tensorstudio._C; x = ts.ones((2, 2)); print(ts.__version__); print(x @ x)"
python -m pip install pytest numpy
pytest -q tests
deactivate
```

## Clean Sdist Test

```powershell
python -m venv .venv-sdist-test
.\.venv-sdist-test\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install dist/tensorstudio-*.tar.gz
python -c "import tensorstudio as ts; import tensorstudio._C; x = ts.ones((2, 2)); print(ts.__version__); print(x + x)"
python -m pip install pytest numpy
pytest -q tests
deactivate
```

## CI Expectations

The Windows GitHub Actions job must import `tensorstudio._C`, run ruff, run
mypy, run pytest, and execute at least one example before a release is promoted.
