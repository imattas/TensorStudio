# Native C++ Core

TensorStudio is designed as a C++ tensor library with Python bindings. Python is
the user-facing API layer, but tensor storage, shape validation, broadcasting,
core math kernels, reductions, autograd metadata, and most differentiable
primitive operations live in C++.

## Native Extension

The extension module is:

```python
import tensorstudio._C
```

It is built from:

- `include/tensorstudio/*.hpp`: public C++ headers.
- `src/core/*.cpp`: tensor runtime and kernels.
- `src/bindings/*.cpp`: pybind11 bindings.

## C++ Responsibilities

The C++ layer owns:

- Tensor storage and dtype conversion.
- Shape, stride, offset, and contiguous-layout logic.
- NumPy-style broadcasting for primitive operations.
- Elementwise arithmetic and comparisons.
- Matrix multiplication.
- Reductions and arg reductions.
- CPU convolution and pooling kernels.
- Reverse-mode autograd graph metadata and backward traversal.
- Error types mapped into Python exceptions.

## Python Responsibilities

Python stays intentionally thin where performance matters:

- Public function names and module organization.
- `nn.Module` composition and model ergonomics.
- Optimizer loops that mutate native tensors.
- Data loading and examples.
- Optional image and ONNX utilities where third-party Python packages are the
  practical integration point.

If a feature performs heavy numeric loops over tensor data, it should usually be
implemented in C++ first.

## Current Native Ops

Native C++ ops include:

- Arithmetic: `add`, `sub`, `mul`, `div`, `neg`, `pow`.
- Comparisons: `equal`, `not_equal`, `less`, `less_equal`, `greater`,
  `greater_equal`.
- Selection: `where`, `maximum`, `minimum`, `clamp`/`clip`.
- Reductions: `sum`, `mean`, `max`, `min`, `argmax`, `argmin`.
- Activations and math: `relu`, `sigmoid`, `tanh`, `exp`, `log`, `log1p`,
  `sqrt`, `rsqrt`, `abs`, trigonometric functions, inverse trigonometric
  functions.
- Layout: `reshape`, `flatten`, `transpose`.
- Vision kernels: `conv2d`, `max_pool2d`, `avg_pool2d`.

## Adding A Native Op

For a new tensor primitive:

1. Add the C++ declaration in `include/tensorstudio/ops.hpp`.
2. Implement the forward kernel in `src/core/ops.cpp`.
3. Add autograd history in the same C++ area when the op is differentiable.
4. Bind it in `src/bindings/bind_ops.cpp` or `bind_tensor.cpp`.
5. Add thin Python exports and type stubs.
6. Test against NumPy where semantics overlap.

