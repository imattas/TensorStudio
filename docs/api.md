# API Overview

This page summarizes the public Python API. The native extension module is
`tensorstudio._C`; most users should import from `tensorstudio`.

## Top-Level Package

```python
import tensorstudio as ts
```

Exports:

- `Tensor`
- `tensor`
- `from_numpy`
- `zeros`
- `ones`
- `full`
- `randn`
- `arange`
- `save`
- `load`
- `nn`
- `optim`
- `__version__`

## Tensor Creation

### `tensor(data, dtype=None, requires_grad=False)`

Creates a tensor from Python numeric scalars, nested sequences, another
TensorStudio tensor, or a NumPy array.

Nested lists must be rectangular. Ragged lists raise a shape error.

### `from_numpy(array, requires_grad=False)`

Creates a TensorStudio tensor copy from a NumPy array.

### Fill And Range Helpers

```python
ts.zeros((2, 3))
ts.ones((2, 3), dtype="float64")
ts.full((2,), 7.0)
ts.randn((4, 4), seed=123)
ts.arange(0, 10, 2)
```

## Tensor Properties

- `Tensor.shape`: tuple of dimensions
- `Tensor.dtype`: dtype string
- `Tensor.ndim`: number of dimensions
- `Tensor.size`: total number of elements
- `Tensor.requires_grad`: whether gradients are tracked
- `Tensor.grad`: accumulated gradient tensor or `None`
- `Tensor.T`: 2D transpose view

## Tensor Methods

Conversion:

- `numpy()`
- `tolist()`
- `item()`

Views:

- `reshape(*shape)` or `reshape(shape)`
- `flatten()`
- `transpose()`
- `T`

Reductions:

- `sum()`
- `mean()`

Activations and math:

- `relu()`
- `sigmoid()`
- `tanh()`
- `exp()`
- `log()`

Autograd:

- `backward(gradient=None)`
- `zero_grad()`

## Operators

TensorStudio overloads common Python operators:

- `x + y`
- `x - y`
- `x * y`
- `x / y`
- `-x`
- `x ** 2`
- `x @ y`

Python scalars and compatible tensor shapes are accepted for elementwise
operations.

## `tensorstudio.nn`

- `Module`
- `Parameter`
- `Linear`
- `Sequential`
- `ReLU`
- `Sigmoid`
- `Tanh`
- `MSELoss`

Modules implement `parameters()`, `zero_grad()`, `train()`, `eval()`, and
`__call__`.

## `tensorstudio.optim`

- `SGD(params, lr=0.01)`
- `Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8)`

Optimizers implement `zero_grad()` and `step()`.

## Error Types

Import native exception aliases from `tensorstudio.errors`:

- `TensorStudioError`
- `ShapeError`
- `DTypeError`
- `DeviceError`
- `AutogradError`

## Serialization

```python
ts.save(obj, "object.tsmodel")
loaded = ts.load("object.tsmodel")
```

Serialization uses pickle in v0.1.x. Only load files from trusted sources.
