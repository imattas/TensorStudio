# macOS

macOS is the third release target after Windows and Linux pass. TensorStudio
should build with Apple Clang and support x86_64 and arm64 wheels where
cibuildwheel runners make that practical.

## Wheel Install

```bash
python -m pip install -U pip
python -m pip install tensorstudio
python -c "import tensorstudio as ts; import tensorstudio._C; print(ts.__version__)"
```

## Source Build

Install Xcode Command Line Tools:

```bash
xcode-select --install
```

Then:

```bash
python -m pip install -U pip
python -m pip install -e ".[dev]"
python -c "import tensorstudio as ts; import tensorstudio._C; print(ts.__version__)"
pytest -q
```

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

## Notes

Avoid compiler flags that assume GCC-specific behavior. The C++ core is expected
to remain portable across MSVC, GCC, Clang, and Apple Clang.
