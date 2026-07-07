# Installation

TensorStudio is distributed as a Python package backed by a native C++20
extension named `tensorstudio._C`.

## Recommended Install

```bash
python -m pip install -U pip
python -m pip install tensorstudio
```

Wheels are the intended install path for normal users. When a matching wheel is
available, installing TensorStudio should not require CMake, a compiler, or
Visual Studio Build Tools.

## Optional Extras

Install ONNX export support:

```bash
python -m pip install "tensorstudio[onnx]"
```

Install Pillow-backed image IO:

```bash
python -m pip install "tensorstudio[vision]"
```

Install both:

```bash
python -m pip install "tensorstudio[onnx,vision]"
```

## Verify The Native Extension

```bash
python - <<'PY'
import tensorstudio as ts
import tensorstudio._C

print(ts.__version__)
print((ts.ones((2, 2)) + 1).tolist())
PY
```

On Windows PowerShell:

```powershell
python -c "import tensorstudio as ts; import tensorstudio._C; print(ts.__version__); print((ts.ones((2, 2)) + 1).tolist())"
```

## Source Builds

If no compatible wheel exists for your platform, pip may build from source.
Source builds require:

- Python 3.10 or newer.
- CMake 3.18 or newer.
- A C++20 compiler.
- pybind11 through the build environment.

See [Build From Source](source-build.md) for platform notes.

