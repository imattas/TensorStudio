# Linux

Linux is the second release target after Windows passes. TensorStudio should
build with GCC and Clang-compatible C++20 toolchains and produce manylinux
wheels through cibuildwheel.

## Wheel Install

```bash
python -m pip install -U pip
python -m pip install tensorstudio
python -c "import tensorstudio as ts; import tensorstudio._C; print(ts.__version__)"
```

## Source Build

Install a compiler and CMake through your distribution packages, then:

```bash
python -m pip install -U pip
python -m pip install -e ".[dev]"
python -c "import tensorstudio as ts; import tensorstudio._C; print(ts.__version__)"
pytest -q
```

## Optional BLAS

TensorStudio source builds can use CBLAS-backed `float32` and `float64`
matrix multiplication when CMake finds a compatible BLAS library and `cblas.h`.
Install a provider such as OpenBLAS, BLIS, or MKL through your distribution or
toolchain.

Examples:

```bash
sudo apt-get install libopenblas-dev
python -m pip install -e ".[dev]"
python -c "import tensorstudio as ts; print(ts.performance_info())"
```

If `blas_enabled` is `False`, TensorStudio is using the portable C++ fallback.
That fallback is tested and intended to keep wheels/source builds portable.

## Build And Check

```bash
python -m build
python -m twine check dist/*
```

## Clean Install Checks

```bash
python -m venv .venv-wheel-test
. .venv-wheel-test/bin/activate
python -m pip install -U pip
python -m pip install dist/*.whl
python -m pip install pytest numpy
pytest -q tests
deactivate
```

```bash
python -m venv .venv-sdist-test
. .venv-sdist-test/bin/activate
python -m pip install -U pip
python -m pip install dist/tensorstudio-*.tar.gz
python -m pip install pytest numpy
pytest -q tests
deactivate
```

## Wheel Notes

The wheel workflow uses cibuildwheel and skips musllinux for this release
candidate. manylinux x86_64 wheels are the primary Linux distribution target.
